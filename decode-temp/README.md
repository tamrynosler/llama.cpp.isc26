# decode-temp/ — Track 2 decode-latency harness

Scripts for measuring single-stream **decode (token-generation) latency** and guarding
output correctness while optimizing the decode path. Single GPU, single node, pinned for
reproducibility. (Track-1 throughput/orchestrator scripts live in `tools/orchestrator/slurm/`.)

## Pinned target

- **Node / GPU:** `xeon-cg2`, GPU 0 — the 80 GB A100 (practice cluster). Chosen so the model
  fit matches the H100's 80 GB; re-run identically on the main cluster tomorrow.
- **Models:** dense `Qwen2.5-14B-Instruct-Q4_0` (primary decode story) + MoE
  `deepseek-moe-16b-base.Q4_K_M` (MoE decode behaves differently — sparse activation).
- GPU is selected via `CUDA_VISIBLE_DEVICES` (no GRES on these clusters); the node is pinned
  with `#SBATCH --nodelist=xeon-cg2` inside the scripts.

## Method

1. **Build** with the optimised flags + CUDA graphs (CUDA graphs default OFF in this fork):
   ```bash
   ml gcc cuda
   decode-temp/build-decode.sh 80      # A100 (xeon-cg2) today;  use 90a on the H100 tomorrow
   ```
2. **Baseline before optimizing**, at the pinned commit. Run both scripts, record numbers +
   reference output. Every later claim is measured against this, on the same node + GPU.
3. **After each optimization**, re-run both on the same node + GPU; compare throughput and
   `diff` the correctness output.

## Scripts

### `build-decode.sh <arch>` — optimised CUDA build
Optimised cluster flags + `-DGGML_CUDA_GRAPHS=ON`. `arch` = `80` (A100) / `90a` (H100).

### `decode-baseline.slurm` — decode throughput
`llama-bench` with `-p 0 -n 128 -d 0,2048,8192 -fa 0,1 -r 5` over both models = a pure decode
test (no prefill) across KV depths, flash-attention off and on. Reports `tg` rows in tok/s
(avg ± stddev); ms/token = 1000 / (t/s).

```bash
sbatch decode-temp/decode-baseline.slurm
```

### `decode-correctness.slurm` — output oracle
Deterministic greedy generation via **`llama-simple`** (greedy argmax, real batch-1 decode
loop, generate-and-exit — no interactive mode, no sampler params to misset) per model. The
generated text is the reference; a decode optimization must reproduce it on the same GPU.
(`llama-cli` is deliberately avoided: it defaults to interactive mode and loops on a batch
job's empty stdin. Perplexity is a fine *quantitative* backup but exercises the batched
prefill path, not the single-token decode kernel — so it's a complement, not the primary gate.)

```bash
sbatch decode-temp/decode-correctness.slurm baseline   # reference (both models)
sbatch decode-temp/decode-correctness.slurm opt1       # after a change
diff decode-correct-dense-baseline.txt decode-correct-dense-opt1.txt
```

### `decode-profile.slurm` — decode profile (L2)
Nsight Systems profile, **both regimes + both models**: dense at d0 / d8192 / d32768 (short →
long context), dense d8192 with CUDA graphs off (launch-overhead control), and MoE at d0 / d8192.
Each writes a `.nsys-rep` (GUI) + `.summary.txt` (kernel ranking, CUDA API/launch, memcpy). Shows
where per-token time goes, how attention grows with KV depth, and how MoE differs from dense.

```bash
sbatch decode-temp/decode-profile.slurm           # -> decode-profile-<jobid>/*.summary.txt
```

### `nsys-shape-breakdown.sh <capture>` — per-matmul-shape decomposition
Un-collapses the GEMV (`mul_mat_vec_q`) by `gridX` (= output rows) so each bucket maps to a
specific matmul (q/k/v/o, FFN gate-up/down, LM head) — the per-shape times give achieved
bandwidth and show bandwidth-bound vs launch/occupancy-bound matmuls. Also dumps flash-attn by
shape. Takes a `.nsys-rep` or `.sqlite`.

```bash
decode-temp/nsys-shape-breakdown.sh decode-profile-<jobid>/dense_d0_graphs_on.nsys-rep
```

### `decode-kvquant.slurm` — KV-cache quantization speedup
Sweeps f16 vs q8_0 vs q4_0 KV cache across KV depths (0→32768) with `-fa 1`, dense model. The
benefit grows with context (smaller cache → fewer bytes read by attention each step). Requires
flash attention (quantized KV only works on the FA path) and a build with
`-DGGML_CUDA_FA_ALL_QUANTS=ON` (build-decode.sh has it). **Lossy** — speed/quality tradeoff, not
output-preserving; quantify quality separately with `llama-perplexity` per KV type.

```bash
sbatch decode-temp/decode-kvquant.slurm     # compare tg at each depth: q8_0/q4_0 vs f16
```

### `decode-perplexity.slurm` — quality / correctness oracle
Runs `llama-perplexity` over wikitext for f16/q8_0/q4_0 KV (quantifies KV-quant's quality cost),
and doubles as the **quantitative correctness oracle** for L3: a decode kernel change should leave
PPL unchanged within tolerance (f16, before vs after). Complements the `llama-simple` greedy diff.

```bash
sbatch decode-temp/decode-perplexity.slurm    # Final estimate: PPL per KV type
```

### `decode-profile-ncu.slurm` — per-kernel drill (L2b)
Nsight Compute on the decode kernels nsys flagged (`mul_mat_vec_q`, `quantize_q8_1`,
`rms_norm_f32`, `flash_attn_ext_f16`). Reports Memory vs Compute throughput (% of peak) and
achieved DRAM GB/s, to decide whether the GEMV is bandwidth-saturated (reduce bytes) or has
headroom (optimize the kernel). Graphs disabled for deterministic launch skip/count.

```bash
sbatch decode-temp/decode-profile-ncu.slurm       # -> decode-ncu-<jobid>/dense_decode.details.txt
```

Outputs/logs write to the launch CWD (not into the repo tree). Copy them back to the parent
`cluster-reports/` per the project workflow.

## H100 main-cluster runbook (turnkey)

The scripts are pinned to `xeon-cg2` (A100 practice). On the H100 cluster, override the node at
submit time (sbatch CLI beats the `#SBATCH --nodelist` in-file) and rebuild for the H100 arch:

```bash
cd ~/llama.cpp.isc26 && git pull
decode-temp/build-decode.sh 90a                                    # H100 = sm_90a (A100 was 80)
H="<h100node>"; P="<partition>"                                    # e.g. cn001 / workers
sbatch --nodelist=$H --partition=$P decode-temp/decode-baseline.slurm        # 1. throughput baseline
sbatch --nodelist=$H --partition=$P decode-temp/decode-correctness.slurm baseline  # 2. golden output
sbatch --nodelist=$H --partition=$P decode-temp/decode-profile.slurm         # 3. nsys (both regimes/models)
# then on a login node:
decode-temp/nsys-shape-breakdown.sh decode-profile-<jobid>/dense_d0_graphs_on.nsys-rep   # 4. GEMV by shape
sbatch --nodelist=$H --partition=$P decode-temp/decode-profile-ncu.slurm     # 5. IF counter perms enabled
```

Notes: paths (`/shared/llms/...`) are shared-FS, same on both clusters. `DATASET=` for perplexity
is overridable inline. The A100 finding — short-context decode is near the memory wall, KV-quant is
a net loss — should be re-confirmed here (HBM3 bandwidth + cc 9.0 kernel-selection thresholds differ).
