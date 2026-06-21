#!/usr/bin/env bash
#SBATCH --job-name=sw-dense-balanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=sw-dense-balanced-%j.log
#
# BALANCED sweep, DENSE models (Qwen2.5-14B-Q8 small, Mistral-Large-2407-Q8 large).
# Equal-ish prompt and generation with a parallel-sequence sweep - the realistic mixed case.
# All knobs are in play; we vary -ub, KV quant, and flash-attn off the base.
#   Usage: sbatch tools/orchestrator/slurm/sweeps/dense-balanced.slurm
set -uo pipefail
REPO="$HOME/llama.cpp.isc26"

SWEEP_NAME="dense-balanced"
MODELS=(qwen14b mistral-large)

# balanced: equal npp/ntg, sweep npl. CTX holds max(npl)*(pp+tg).
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
