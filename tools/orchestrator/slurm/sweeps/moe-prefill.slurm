#!/usr/bin/env bash
#SBATCH --job-name=sw-moe-prefill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=sw-moe-prefill-%j.log
#
# PREFILL-HEAVY sweep, MoE models (deepseek-moe-16b-Q8 small, Llama-4-Scout-17B-16E-Q8 large).
# MoE has a large memory footprint but small active compute, so the prefill/decode balance and
# the split economics differ from dense - same knobs, expect a different optimum.
#   Usage: sbatch tools/orchestrator/slurm/sweeps/moe-prefill.slurm
set -uo pipefail
REPO="$HOME/llama.cpp.isc26"

SWEEP_NAME="moe-prefill"
MODELS=(deepseek-moe scout)

PP_small="2048,4096,8192"; TG_small="32"; PL_small="1,4";   CTX_small=40960
PP_large="2048,4096";      TG_large="32"; PL_large="1,2";   CTX_large=12288

KNOBS=(
  "base|-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"
  "ub512|-b 4096 -ub 512 -fa on -ctk f16 -ctv f16"
  "ub4096|-b 4096 -ub 4096 -fa on -ctk f16 -ctv f16"
  "b8192|-b 8192 -ub 2048 -fa on -ctk f16 -ctv f16"
  "fa_off|-b 4096 -ub 2048 -fa off -ctk f16 -ctv f16"
)
BASELINE_KNOBS="-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"

source "$REPO/tools/orchestrator/slurm/sweeps/_sweep_common.sh"
run_sweep
