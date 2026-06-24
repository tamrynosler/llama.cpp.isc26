#!/bin/bash
#SBATCH --job-name=tune-t4
#SBATCH --output=tune-t4-%j.out.txt
#SBATCH --error=tune-t4-%j.err.txt
#SBATCH --partition=workers
#SBATCH --nodes=1
#SBATCH --nodelist=xeon-cg2
#
# T-4 single-GPU parameter sweep: 70B Q4_K_M on one 80 GB A100. Workload 49152 / 2048.
# THE INTERESTING / 2x-WEIGHTED TIER: big prefill AND big generation.
#
# Binding levers = q8_0 KV + B (-npl) + -ub, and they compete for memory:
#   - q8_0 KV cache (-ctk/-ctv q8_0, ~half of f16) frees VRAM to raise B, which lifts the decode
#     term of the score at low batch.
#   - a larger -ub lifts the prefill term.
#   - both eat the same budget, so this is a tradeoff sweep -- find the (npl, ub) that maximizes
#     overall S t/s.
#
# ctx fixed at 153600 = npl_max(3) x (npp+ntg)=51200, q8_0 KV (~26 GB) on top of ~43 GB weights.
# npl=4 would need a larger ctx whose KV + prefill compute buffer overflows 80 GB, so we cap at 3
# and let the sweep show where it OOMs. -fa on is REQUIRED to quantize the V cache.
# -ub values run as separate invocations so an OOM doesn't abort the smaller combos.
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
    echo "############################  -ub ${UB}  (q8_0 KV)  ############################"
    "$BIN" -m "$M" -fa on -ngl 999 -ctk q8_0 -ctv q8_0 -c 153600 -b 4096 -ub "$UB" \
        -npp 49152 -ntg 2048 -npl 1,2,3 || echo "(-ub ${UB} aborted -- likely OOM; try fewer npl or smaller ub)"
done
