#!/usr/bin/env bash
#SBATCH --job-name=sw-dense-decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=sw-dense-decode-%j.log
#
# DECODE-HEAVY sweep, DENSE models (Qwen2.5-14B-Q8 small, Mistral-Large-2407-Q8 large).
# Tiny prompt, long generation -> memory-bandwidth-bound decode. The levers are the parallel
# sequence count (npl, in the shapes) and KV-cache quant (-ctk/-ctv); -ub barely matters.
#   Usage: sbatch tools/orchestrator/slurm/sweeps/dense-decode.slurm
set -uo pipefail
REPO="$HOME/llama.cpp.isc26"

SWEEP_NAME="dense-decode"
MODELS=(qwen14b mistral-large)

# decode-heavy: tiny npp, long ntg, npl is the throughput lever. CTX must hold max(npl)*(pp+tg).
PP_small="128"; TG_small="1024,2048"; PL_small="1,8,32"; CTX_small=73728
PP_large="128"; TG_large="512,1024";  PL_large="1,4";    CTX_large=8192

# Phase A: decode emphasises KV-cache quant and flash-attn.
KNOBS=(
  "base|-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"
  "kv_q8|-b 4096 -ub 2048 -fa on -ctk q8_0 -ctv q8_0"
  "fa_off|-b 4096 -ub 2048 -fa off -ctk f16 -ctv f16"
  "ub512|-b 4096 -ub 512 -fa on -ctk f16 -ctv f16"
)
BASELINE_KNOBS="-b 4096 -ub 2048 -fa on -ctk f16 -ctv f16"

source "$REPO/tools/orchestrator/slurm/sweeps/_sweep_common.sh"
run_sweep
