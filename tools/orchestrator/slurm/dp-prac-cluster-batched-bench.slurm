#!/bin/bash
#SBATCH --job-name=dp-prac-cluster-bb
#SBATCH --output=dp-prac-cluster-bb-%j.out.txt
#SBATCH --error=dp-prac-cluster-bb-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=3
#SBATCH --nodelist=xeon-cg1,xeon-cg2,xeon-cg3
#SBATCH --ntasks-per-node=1
#
# Multi-node data-parallel batched-bench (launcher fan-out, PRACTICE cluster: 3 nodes x 2 A100 = 6).
#
# Practice-cluster twin of dp-cluster-batched-bench.slurm. srun starts one task per node; each runs
# the STOCK llama-batched-bench with --data-parallel 2 across its 2 local A100s (node-local library
# DP, batched-bench.cpp unmodified) and writes its own JSONL. The nodes do NOT communicate during
# inference (weak scaling), so cluster throughput per config is the SUM of the per-node throughput;
# aggregate-cluster.py reports that and the node-scaling (ideal = 3).
#
# Build first (A100 = sm_80), on feat/orchestrator2:
#   ml gcc; ml cuda
#   cmake -B build-a100 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-a100 --target llama-batched-bench -j$(nproc)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

# ---- perf-per-watt throttle (mirrors isc26 maxout): CPUs low + GPUs capped to 200 W ----
# Uses the isc26 power helpers (ssh-loop over $SLURM_JOB_NODELIST), reached via POWER_DIR (default
# $HOME/isc26/power). CPU governors are restored on any exit; the per-rank wrapper on the workload
# below re-raises the working rank's CPUs. GPU cap is left set on exit, same as maxout.
export POWER="${POWER_DIR:-$HOME/isc26/power}"
trap "$POWER/reset_nodes_cpus.sh" EXIT INT TERM
"$POWER/set_nodes_cpus_low.sh"
"$POWER/set_nodes_gpus_pl.sh" 200

export REPO="$HOME/llama.cpp.isc26"
export BIN="$REPO/build-a100/bin/llama-batched-bench"
export M="/shared/llms/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
export RESULTS="$HOME/cluster-results/${SLURM_JOB_ID:-local}"   # shared FS, kept out of the repo
mkdir -p "$RESULTS"

export NGL=99 PP=512 TG=128 PL="1,2,4,8,16" CTX=16384 NBATCH=2048 UBATCH=512

# one task per node; vars above are inherited via srun's default --export=ALL and evaluated ON the
# node (single-quoted body), so each node pins its own 2 GPUs and writes its own JSONL to stdout.
srun --ntasks-per-node=1 bash -c '
  export CUDA_VISIBLE_DEVICES=0,1
  "$POWER/set_rank_cpus_high.sh" "$BIN" -m "$M" -ngl "$NGL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" \
    -npp "$PP" -ntg "$TG" -npl "$PL" --data-parallel 2 --output-format jsonl \
    > "$RESULTS/node-$(hostname).jsonl"
'

echo "per-node JSONL written to $RESULTS"
echo ""
python3 "$REPO/tools/orchestrator/slurm/aggregate-cluster.py" "$RESULTS"
