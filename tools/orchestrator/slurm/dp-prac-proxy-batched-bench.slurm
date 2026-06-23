#!/bin/bash
#SBATCH --job-name=dp-prac-proxy-bb
#SBATCH --output=dp-prac-proxy-bb-%j.out.txt
#SBATCH --error=dp-prac-proxy-bb-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# orchestrator2 single-node data-parallel smoke + scaling check (PRACTICE cluster, A100).
#
# Practice-cluster twin of dp-proxy-batched-bench.slurm. Runs the STOCK llama-batched-bench
# (byte-identical to upstream) twice on one practice node (2x A100): once with no DP (the 1-GPU
# baseline) and once with --data-parallel 2. The DP run loads two independent replicas (one per
# A100) inside the library and shards each decode's parallel sequences across them. Compare the
# two tables: the -dp 2 ALL throughput at pl >= 2 should approach 2x the baseline, with
# batched-bench.cpp unmodified.
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
BIN="$REPO/build-a100/bin/llama-batched-bench"
M="/shared/llms/models/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"

# both A100s on the node visible; DP pins replica r -> device r.
export CUDA_VISIBLE_DEVICES=0,1

NGL=99
PP=512
TG=128
PL="1,2,4,8,16"     # pl=1 cannot shard; higher pl => more seqs/GPU => closer to 2x
CTX=16384           # >= pl*(pp+tg) for the largest pl (16*640=10240)
NBATCH=2048
UBATCH=512

COMMON=(-m "$M" -ngl "$NGL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -npp "$PP" -ntg "$TG" -npl "$PL")

echo "################  BASELINE: single GPU (no data parallelism)  ################"
CUDA_VISIBLE_DEVICES=0 "$BIN" "${COMMON[@]}"

echo ""
echo "################  DATA-PARALLEL: --data-parallel 2 (both A100s)  ################"
"$BIN" "${COMMON[@]}" --data-parallel 2
