#!/bin/bash
#SBATCH --job-name=tune-t2
#SBATCH --output=tune-t2-%j.out.txt
#SBATCH --error=tune-t2-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# T-2 single-GPU parameter sweep: 70B Q4_K_M on one 80 GB A100. Workload 4096 / 1024.
#
# Binding lever = B (-npl) + -ub. Decode share benefits from batching (raise B); the 4k prefill
# share benefits from a larger ubatch (better tensor-core utilization). Sweep both and read the
# S t/s / S_PP / S_TG columns to find the joint optimum.
#
# ctx fixed at 81920 = npl_max(16) x (npp+ntg)=5120. f16 KV. -b 4096 (full sequence per submit).
# The -ub values run as separate invocations so an OOM at the largest (npl,ub) combo doesn't abort
# the smaller ones (completed rows print incrementally regardless).
#
# Build: see tune-t1-decode.slurm.

set -euo pipefail
source /etc/profile.d/lmod.sh
module purge
module load gcc
module load cuda

BIN="$HOME/llama.cpp.isc26/build-a100/bin/llama-batched-bench"
M="/shared/llms/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
export CUDA_VISIBLE_DEVICES=0

for UB in 512 1024 2048; do
    echo ""
    echo "############################  -ub ${UB}  ############################"
    "$BIN" -m "$M" -fa on -ngl 999 -c 81920 -b 4096 -ub "$UB" \
        -npp 4096 -ntg 1024 -npl 1,2,4,8,16 || echo "(-ub ${UB} aborted early -- likely OOM at the largest npl)"
done
