#!/bin/bash
#SBATCH --job-name=dp-cluster-bb
#SBATCH --output=dp-cluster-bb-%j.out.txt
#SBATCH --error=dp-cluster-bb-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=3
#SBATCH --nodelist=cn001,cn002,cn003
#SBATCH --ntasks-per-node=1
#
# Multi-node data-parallel batched-bench (launcher fan-out, H100 cluster: 3 nodes x 4 H100 = 12).
#
# srun starts one task per node; each task runs the STOCK llama-batched-bench with
# --data-parallel 4 across its 4 local H100s (node-local library DP, batched-bench.cpp unmodified)
# and writes its own JSONL. The nodes do NOT communicate during inference (weak scaling), so
# cluster throughput per config is simply the SUM of the per-node throughput. aggregate-cluster.py
# does that sum and reports node-scaling (ideal = 3).
#
# Build first (H100 = sm_90a), on feat/orchestrator2:
#   ml gcc; ml cuda
#   cmake -B build-h100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90a -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-h100 --target llama-batched-bench -j$(nproc)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

export REPO="$HOME/llama.cpp.isc26"
export BIN="$REPO/build-h100/bin/llama-batched-bench"
export M="/shared/llms/models/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"
export RESULTS="$HOME/cluster-results/${SLURM_JOB_ID:-local}"   # shared FS, kept out of the repo
mkdir -p "$RESULTS"

export NGL=99 PP=512 TG=128 PL="1,2,4,8,16,32" CTX=32768 NBATCH=2048 UBATCH=512

# one task per node; vars above are inherited via srun's default --export=ALL and evaluated ON the
# node (single-quoted body), so each node pins its own 4 GPUs and writes its own JSONL to stdout.
srun --ntasks-per-node=1 bash -c '
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  "$BIN" -m "$M" -ngl "$NGL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" \
    -npp "$PP" -ntg "$TG" -npl "$PL" --data-parallel 4 --output-format jsonl \
    > "$RESULTS/node-$(hostname).jsonl"
'

echo "per-node JSONL written to $RESULTS"
echo ""
python3 "$REPO/tools/orchestrator/slurm/aggregate-cluster.py" "$RESULTS"
