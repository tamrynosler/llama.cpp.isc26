#!/bin/bash
#SBATCH --job-name=dp-proxy-bb
#SBATCH --output=dp-proxy-bb-%j.out.txt
#SBATCH --error=dp-proxy-bb-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=cn001
#
# orchestrator2 single-node data-parallel smoke + scaling check (H100 cluster).
#
# Runs the STOCK llama-batched-bench (byte-identical to upstream) twice on one node (4x H100):
# once with no DP (the 1-GPU baseline) and once with --data-parallel 4. The DP run loads four
# independent replicas (one per H100) inside the library and shards each decode's parallel
# sequences across them. Compare the two tables: the -dp 4 ALL throughput at pl >= 4 should
# approach 4x the baseline, with batched-bench.cpp unmodified.
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
BIN="$REPO/build-h100/bin/llama-batched-bench"
M="/shared/llms/models/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"

# all four H100s on the node visible; DP pins replica r -> device r.
export CUDA_VISIBLE_DEVICES=0,1,2,3

NGL=99
PP=512
TG=128
PL="1,2,4,8,16,32"  # pl=1 cannot shard; higher pl => more seqs/GPU => closer to 4x
CTX=32768           # >= pl*(pp+tg) for the largest pl (32*640=20480)
NBATCH=2048
UBATCH=512

COMMON=(-m "$M" -ngl "$NGL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -npp "$PP" -ntg "$TG" -npl "$PL")

echo "################  BASELINE: single GPU (no data parallelism)  ################"
CUDA_VISIBLE_DEVICES=0 "$BIN" "${COMMON[@]}"

echo ""
echo "################  DATA-PARALLEL: --data-parallel 4 (all four H100s)  ################"
"$BIN" "${COMMON[@]}" --data-parallel 4
