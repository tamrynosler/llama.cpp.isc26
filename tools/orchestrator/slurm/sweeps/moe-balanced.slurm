#!/usr/bin/env bash
#SBATCH --job-name=sw-moe-balanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=sw-moe-balanced-%j.log
#
# BALANCED sweep, MoE models (deepseek-moe-16b-Q8 small, Llama-4-Scout-17B-16E-Q8 large).
# Equal-ish prompt/generation with a parallel-sequence sweep - the realistic mixed case for MoE.
#   Usage: sbatch tools/orchestrator/slurm/sweeps/moe-balanced.slurm
set -uo pipefail
REPO="$HOME/llama.cpp.isc26"

SWEEP_NAME="moe-balanced"
MODELS=(deepseek-moe scout)

PP_small="512"; TG_small="512"; PL_small="1,4,16,32"; CTX_small=32768
PP_large="512"; TG_large="512"; PL_large="1,4";       CTX_large=8192

KNOBS=(
  "base|-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"
  "ub512|-b 4096 -ub 512 -fa on -ctk f16 -ctv f16"
  "kv_q8|-b 4096 -ub 2048 -fa on -ctk q8_0 -ctv q8_0"
  "fa_off|-b 4096 -ub 2048 -fa off -ctk f16 -ctv f16"
)
BASELINE_KNOBS="-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"

source "$REPO/tools/orchestrator/slurm/sweeps/_sweep_common.sh"
run_sweep
