#!/usr/bin/env bash
#SBATCH --job-name=decode-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-profile-%j.log
#
# Track 2 (L2) — decode-phase profile with Nsight Systems (nsys), both regimes + both models.
#
# Captures (one config each, so every timeline/summary is attributable):
#   dense d0    graphs on  — short-context decode kernel mix (GEMV-dominated, near memory wall)
#   dense d8192 graphs on  — mid-context: attention/KV starts to grow
#   dense d32768 graphs on — LONG-context regime: attention/KV becomes a major share
#   dense d8192 graphs off — kernel-launch overhead that CUDA graphs remove (control)
#   moe   d0    graphs on  — MoE short-context (expert routing differs from dense)
#   moe   d8192 graphs on  — MoE mid-context
#
# Each capture writes a .nsys-rep (GUI) + .summary.txt (kernel/API/memcpy tables). Then run
# decode-temp/nsys-shape-breakdown.sh on a capture for the per-matmul-shape decomposition.
#
# Pinned to xeon-cg2 GPU 0. For the H100 main cluster, override at submit time, e.g.:
#   sbatch --nodelist=<h100node> --partition=<part> decode-temp/decode-profile.slurm
# (and build with build-decode.sh 90a). nsys ships with CUDA; if missing try `ml nsight-systems`.
#
# Usage:  sbatch decode-temp/decode-profile.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-bench"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
MOE="/shared/llms/models/deepseek-moe-16b-base.Q4_K_M.gguf"
GPU_IDX=0
NGEN=128             # decode steps to profile (plenty for the CUDA graph to warm up + steady state)
OUTDIR="decode-profile-${SLURM_JOB_ID:-manual}"

# name                   model  depth  graphs(on|off)
PROFILES=(
  "dense_d0_graphs_on     DENSE  0      on"
  "dense_d8192_graphs_on  DENSE  8192   on"
  "dense_d32768_graphs_on DENSE  32768  on"
  "dense_d8192_graphs_off DENSE  8192   off"
  "moe_d0_graphs_on       MOE    0      on"
  "moe_d8192_graphs_on    MOE    8192   on"
)
# -----------------------------------------------------------------------------

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not on PATH. Try:  ml nsight-systems  (or the site's Nsight Systems module)" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD)" >&2
nsys --version >&2

for cfg in "${PROFILES[@]}"; do
    read -r name model_var depth graphs <<< "$cfg"
    model="${!model_var}"            # indirect: DENSE / MOE -> path
    rep="$OUTDIR/$name"
    pre=()
    [ "$graphs" = "off" ] && pre=(env GGML_CUDA_DISABLE_GRAPHS=1)
    echo ">>> $name (model=$model_var depth=$depth graphs=$graphs)" >&2
    "${pre[@]}" nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --force-overwrite true \
        -o "$rep" \
        "$BIN" -m "$model" -ngl 99 -p 0 -n "$NGEN" -d "$depth" -fa 1 -r 1 --no-warmup -o md
    nsys stats --force-export=true "$rep.nsys-rep" > "$rep.summary.txt" 2>&1
    echo "    -> $rep.nsys-rep + $rep.summary.txt" >&2
done

echo "done -> $OUTDIR/  (paste *.summary.txt; run nsys-shape-breakdown.sh on a capture)" >&2
