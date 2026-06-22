# Throughput tuning sweeps

Six `sbatch` sweeps that find the best batched-bench config for three workload profiles across
two model families:

|            | prefill-heavy        | decode-heavy        | balanced              |
|------------|----------------------|---------------------|-----------------------|
| **dense**  | `dense-prefill.slurm`| `dense-decode.slurm`| `dense-balanced.slurm`|
| **MoE**    | `moe-prefill.slurm`  | `moe-decode.slurm`  | `moe-balanced.slurm`  |

Each sweep runs **two models** — one that fits a single 80 GB H100 (the DP / oversubscription
story) and one too big for one card (the split / capacity story):

| class   | dense              | MoE                          | ~Q8 footprint | layouts exercised                |
|---------|--------------------|------------------------------|---------------|----------------------------------|
| `small` | Qwen2.5-14B-Q8     | deepseek-moe-16b-Q8          | ~15–17 GB     | 1-GPU → DP2 → DP4 → oversub 2/4  |
| `large` | Mistral-Large-2407-Q8 | Llama-4-Scout-17B-16E-Q8  | ~116–130 GB   | 2-GPU split (1/2 reps) × {row,layer}, 4-GPU split |

All models are Q8 (for cross-model comparability) and expected under `~/models`. A model that
isn't found is skipped with a message — so e.g. the MoE sweeps run Scout and skip deepseek until
its Q8 download lands in `~/models`.

## Design: two phases per (model, workload)

The driver (`_sweep_common.sh`) avoids a full cross-product (which would be thousands of runs).
Instead, **coordinate descent**:

- **Phase A — knob tuning.** At a fixed *reference layout* (small: 1 GPU; large: a 2-GPU row
  split), vary **one knob at a time** off a baseline — `-b`, `-ub`, `-fa` (flash-attn),
  `-ctk`/`-ctv` (KV-cache quant). Prefill sweeps emphasise `-b`/`-ub`; decode sweeps emphasise
  KV quant + flash-attn. This finds each knob's sweet spot at linear cost.
- **Phase B — layout scaling.** At the *baseline knobs*, sweep the orchestration / split layouts
  (DP replica count, oversubscription, split mode). This is the near-linear-scaling result.

The workload itself (prefill vs decode vs balanced) is set by the `-npp`/`-ntg`/`-npl` shapes,
and `-npl` (parallel sequences per replica) is swept inside every run as the main decode lever.

> **Feed-forward note.** Phase B uses the baseline knobs, not Phase A's winner (a single batch
> job can't read its own mid-run results). After Phase A identifies a better knob, optionally
> edit `BASELINE_KNOBS` in that sweep script and re-run Phase B alone for the final numbers.

## Run

```bash
# build the fork first (on a build node): cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
sbatch tools/orchestrator/slurm/sweeps/dense-balanced.slurm     # start with a small-fast one to validate
# then the rest, e.g.:
for s in dense-prefill dense-decode moe-prefill moe-decode moe-balanced; do
  sbatch tools/orchestrator/slurm/sweeps/$s.slurm
done
```

Results land in `results/sweep-<name>-<jobid>/` as one `<model>.<phase>.<variant>.jsonl` per
config (machine-readable; each line is a pp×tg×npl cell). The job log echoes the command and the
last few jsonl lines of each config for liveness.

**Cost.** Each sweep is ~20+ multi-shape configs over two models; the large models are slow.
Run the small-model-heavy sweeps first to validate, and trim `KNOBS`/`PL_*` at the top of a
script if a job is over its `--time`. Each sweep is an independent job, so they queue in parallel.

## Analyse

Each `.jsonl` line carries the per-config throughput. Pull the headline number per cell, e.g.:

```bash
cd results/sweep-dense-balanced-<jobid>
# dump key fields from every config (adjust field names to your jsonl schema)
for f in *.jsonl; do jq -c --arg cfg "$f" '{cfg:$cfg, n_pp, n_tg, n_pl, t_s:(.S_e2e // .s_e2e // .speed)}' "$f"; done
```

Compare:
- **Phase A:** within a model, the knob variant with the best throughput at the reference layout
  → the per-workload winning config.
- **Phase B:** the DP / split / oversubscription scaling curve at the baseline knobs → the
  near-linear-throughput result (small) and the split-mode trade-off (large, row vs layer).

Integrity: DP / oversubscription rows are **N-replica aggregates** — label them as such, never
as a single-GPU or single-request number.
