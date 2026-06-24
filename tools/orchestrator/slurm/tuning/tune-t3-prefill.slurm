#!/bin/bash
#SBATCH --job-name=tune-t3
#SBATCH --output=tune-t3-%j.out.txt
#SBATCH --error=tune-t3-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
# Either 80 GB A100 node works. To use the other one, override on the command line (wins over the
# line above):   sbatch --nodelist=xeon-cg3 <this script>
#
# T-3 single-GPU parameter sweep: 70B Q4_K_M on one 80 GB A100. Workload 32768 / 512 (prefill-bound).
#
# Binding lever = -ub. Prefill is a compute-saturated GEMM already at B=1, so B barely moves the
# score (S_total -> S_pp). What moves it is a larger ubatch (tensor-core utilization) + flash
# attention. Sweep -ub and watch S_PP rise, until the prefill compute buffer eats the KV budget
# (OOM). npl held small (1,2) just to confirm B is a no-op here.
#
# ctx fixed at 66560 = npl_max(2) x (npp+ntg)=33280. f16 KV. -b 4096.
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
    "$BIN" -m "$M" -fa on -ngl 999 -c 66560 -b 4096 -ub "$UB" \
        -npp 32768 -ntg 512 -npl 1,2 || echo "(-ub ${UB} aborted -- likely prefill compute buffer OOM)"
done
