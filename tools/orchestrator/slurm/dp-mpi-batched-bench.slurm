#!/bin/bash
#SBATCH --job-name=dp-mpi-bb
#SBATCH --output=dp-mpi-bb-%j.out.txt
#SBATCH --error=dp-mpi-bb-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=3
#SBATCH --nodelist=cn001,cn002,cn003
#SBATCH --ntasks-per-node=1
#
# In-source MPI multi-node batched-bench (H100 cluster: 3 nodes x 4 H100 = 12 GPUs).
#
# srun starts one rank per node. The library (built with -DLLAMA_DP_MPI=ON) makes rank 0 the driver
# and ranks 1-2 MPI compute workers: ONE stock llama-batched-bench process (rank 0) shards every
# decode across all 12 GPUs and prints a SINGLE table with cluster-wide throughput. The other ranks
# never run the benchmark's main. batched-bench.cpp is unmodified.
#
# Contrast with dp-cluster-batched-bench.slurm (launcher): that runs 3 independent processes and
# sums their throughput externally; this is one unified run. Same total throughput, one table.
#
# Build first (needs MPI + CUDA, sm_90a), on feat/orchestrator2:
#   ml gcc; ml cuda; ml hpcx
#   cmake -B build-h100-mpi -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90a -DLLAMA_DP_MPI=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-h100-mpi --target llama-batched-bench -j$(nproc)

set -euo pipefail

source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda
module load hpcx

# ---- perf-per-watt throttle (mirrors isc26 maxout): CPUs low + GPUs capped to 200 W ----
# Uses the isc26 power helpers (ssh-loop over $SLURM_JOB_NODELIST), reached via POWER_DIR (default
# $HOME/isc26/power). CPU governors are restored on any exit; the per-rank wrapper on the workload
# below re-raises the working rank's CPUs. GPU cap is left set on exit, same as maxout.
export POWER="${POWER_DIR:-$HOME/isc26/power}"
trap "$POWER/reset_nodes_cpus.sh" EXIT INT TERM
"$POWER/set_nodes_cpus_low.sh"
"$POWER/set_nodes_gpus_pl.sh" 200

REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-h100-mpi/bin/llama-batched-bench"
M="/shared/llms/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"

# each rank pins its node's 4 H100s; the library shards across all ranks via MPI.
export CUDA_VISIBLE_DEVICES=0,1,2,3

NGL=99
PP=512
TG=128
PL="1,2,4,8,16,32"   # global pl sharded across all 12 GPUs (R = 3 nodes x 4 = 12)
CTX=32768
NBATCH=2048
UBATCH=512

# one rank per node (pmix); --data-parallel 4 = local replicas/node. rank 0 prints the table.
srun --mpi=pmix --ntasks-per-node=1 "$POWER/set_rank_cpus_high.sh" "$BIN" \
    -m "$M" -ngl "$NGL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" \
    -npp "$PP" -ntg "$TG" -npl "$PL" --data-parallel 4
