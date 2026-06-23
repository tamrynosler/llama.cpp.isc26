#!/bin/bash
#SBATCH --job-name=dp-prac-proxy-ppl
#SBATCH --output=dp-prac-proxy-ppl-%j.out.txt
#SBATCH --error=dp-prac-proxy-ppl-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# orchestrator2 perplexity correctness + speedup check (PRACTICE cluster, A100).
#
# Practice-cluster twin of dp-proxy-perplexity.slurm. Runs the STOCK llama-perplexity twice on
# one practice node (2x A100): once single-GPU (the baseline PPL) and once with --data-parallel 2.
# The proxy shards perplexity's parallel chunks across the two replicas and reads each token's
# logits back from the replica that produced it.
#
# Expected: the DP "Final estimate: PPL" matches the baseline PPL (per-sequence logits identical),
# so the <=10% gate is met by construction; the DP pass is faster (chunks run concurrently).
#
# -c 512 (chunk) with -b 4096 => n_seq = 8 (4 sequences per replica on 2 GPUs).
#
# Build first (A100 = sm_80), from the repo root, on feat/orchestrator2:
#   ml gcc; ml cuda
#   cmake -B build-a100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-a100 -j$(nproc)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-a100/bin/llama-perplexity"
M="/shared/llms/models/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"

# wikitext-2 raw test split. Adjust if the dataset lives elsewhere on the cluster.
#   https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
DATA="${WIKITEXT:-/shared/llms/datasets/wikitext-2-raw/wiki.test.raw}"

export CUDA_VISIBLE_DEVICES=0,1

NGL=99
CHUNK=512
NBATCH=4096

COMMON=(-m "$M" -f "$DATA" -ngl "$NGL" -c "$CHUNK" -b "$NBATCH")

echo "################  BASELINE PPL: single GPU (no data parallelism)  ################"
CUDA_VISIBLE_DEVICES=0 "$BIN" "${COMMON[@]}"

echo ""
echo "################  DATA-PARALLEL PPL: --data-parallel 2 (both A100s)  ################"
"$BIN" "${COMMON[@]}" --data-parallel 2
