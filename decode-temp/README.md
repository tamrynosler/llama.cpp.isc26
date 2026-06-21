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
Nsight Systems profile of the **dense** decode loop: three captures — depth 0 / depth 8192 /
depth 8192 with CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`). Each writes a `.nsys-rep`
(GUI) + a `.summary.txt` (kernel-time ranking, CUDA API/launch time, memcpy). Answers where
per-token time goes, how attention cost grows with KV depth, and how much launch overhead
CUDA graphs already remove. `ncu` per-kernel memory-bandwidth analysis is the follow-up once
nsys names the dominant kernels.

```bash
sbatch decode-temp/decode-profile.slurm           # -> decode-profile-<jobid>/*.summary.txt
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

> **Assumed path:** the MoE model dir is taken to be `/shared/llms/models/` (only the dense
> model's full path was given). Fix `MOE=` in both `.slurm` files if it lives elsewhere.
