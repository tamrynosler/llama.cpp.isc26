#!/bin/bash
#SBATCH --job-name=dp-proxy-ppl
#SBATCH --output=dp-proxy-ppl-%j.out.txt
#SBATCH --error=dp-proxy-ppl-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=cn001
#
# orchestrator2 perplexity correctness + speedup check (H100 cluster).
#
# Runs the STOCK llama-perplexity (byte-identical to upstream) twice on one node (4x H100):
# once single-GPU (the baseline PPL) and once with --data-parallel 4. perplexity processes
# n_seq = n_batch/chunk chunks as parallel sequences; the proxy shards those sequences across
# the four replicas and reads each token's logits back from the replica that produced it.
#
# Expected: the DP "Final estimate: PPL" matches the baseline PPL (each sequence runs end-to-end
# on one full replica, so per-sequence logits are identical) -> the <=10% gate is met by
# construction, and the DP pass is faster because the chunks run concurrently.
#
# -c 512 (chunk) with -b 4096 => n_seq = 8 (2 sequences per replica on 4 GPUs).
#
# Build first (H100 = sm_90a), from the repo root, on feat/orchestrator2:
#   ml gcc; ml cuda
#   cmake -B build-h100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90a -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-h100 --target llama-batched-bench llama-perplexity -j$(nproc)
#   (target only the two binaries we use; building everything pulls in tools/ui, which fails to
#    compile on nodes without npm/internet for the UI asset download)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-h100/bin/llama-perplexity"
M="/shared/llms/models/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"

# wikitext-2 raw test split. Adjust if the dataset lives elsewhere on the cluster.
#   https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
DATA="${WIKITEXT:-/shared/llms/datasets/wikitext-2-raw/wiki.test.raw}"

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGL=99
CHUNK=512
NBATCH=4096

COMMON=(-m "$M" -f "$DATA" -ngl "$NGL" -c "$CHUNK" -b "$NBATCH")

echo "################  BASELINE PPL: single GPU (no data parallelism)  ################"
CUDA_VISIBLE_DEVICES=0 "$BIN" "${COMMON[@]}"

echo ""
echo "################  DATA-PARALLEL PPL: --data-parallel 4 (all four H100s)  ################"
"$BIN" "${COMMON[@]}" --data-parallel 4
