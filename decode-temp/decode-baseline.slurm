#!/usr/bin/env bash
#SBATCH --job-name=decode-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-baseline-%j.log
#
# Track 2 (decode-phase latency) — STOCK single-stream decode baseline.
#
# Measures token-generation (decode) throughput with llama-bench on ONE pinned GPU.
# This is the number every Track-2 optimization is compared against: capture it on the
# pinned commit BEFORE touching any code, then re-run on the SAME node + GPU after each
# change. The `tg` rows are the decode metric; ms/token = 1000 / (t/s).
#
# Pinned to: node xeon-cg2, GPU 0 (the 80 GB A100 — matches the H100's 80 GB so the model
# fit is identical when this is re-run on the main cluster). No GRES; GPU is selected via
# CUDA_VISIBLE_DEVICES. Build first with decode-temp/build-decode.sh 80 (A100) / 90a (H100).
#
# Usage:  sbatch decode-temp/decode-baseline.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-bench"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
MOE="/shared/llms/models/deepseek-moe-16b-base.Q4_K_M.gguf"   # NOTE: dir assumed = /shared/llms/models
GPU_IDX=0            # GPU 0 on xeon-cg2 = 80 GB A100
DEPTHS="0,2048,8192" # decode at increasing KV depth — decode slows as the cache grows
NGEN=128             # tokens generated per decode test
REPS=5               # repetitions; llama-bench reports avg +/- stddev
# -----------------------------------------------------------------------------

ml gcc cuda

export CUDA_VISIBLE_DEVICES="$GPU_IDX"   # expose exactly one GPU -> llama-bench main_gpu 0

echo "=== provenance ==="
echo "node   = $(hostname)"
echo "commit = $(git -C "$REPO" rev-parse --short HEAD)"
echo "gpu_idx= $GPU_IDX (CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=================="

# Pure decode test: -p 0 skips the prefill test; -n generates; -d sweeps KV depth;
# -fa 0,1 runs flash-attention off AND on (FA is a decode-relevant kernel).
# Both models run in one invocation (llama-bench loads each -m in turn).
"$BIN" -m "$DENSE" -m "$MOE" -ngl 99 -p 0 -n "$NGEN" -d "$DEPTHS" -fa 0,1 -r "$REPS" -o md
