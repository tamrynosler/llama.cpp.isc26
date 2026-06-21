#!/usr/bin/env bash
#SBATCH --job-name=sw-dense-prefill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=sw-dense-prefill-%j.log
#
# PREFILL-HEAVY sweep, DENSE models (Qwen2.5-14B-Q8 small, Mistral-Large-2407-Q8 large).
# Long prompts, tiny generation -> compute-bound prefill. The knobs that matter here are the
# logical/physical batch (-b/-ub) and flash-attn; KV quant barely moves a prefill number.
#   Usage: sbatch tools/orchestrator/slurm/sweeps/dense-prefill.slurm
set -uo pipefail
REPO="$HOME/llama.cpp.isc26"

SWEEP_NAME="dense-prefill"
MODELS=(qwen14b mistral-large)

# prefill-heavy: big npp, tiny ntg. small fits big shapes; large kept lighter (2-GPU KV budget).
PP_small="2048,4096,8192"; TG_small="32"; PL_small="1,4";   CTX_small=40960
PP_large="2048,4096";      TG_large="32"; PL_large="1,2";   CTX_large=12288

# Phase A: vary one knob off base. Prefill emphasises -ub / -b / flash-attn.
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
