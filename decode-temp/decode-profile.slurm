#!/usr/bin/env bash
#SBATCH --job-name=decode-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-profile-%j.log
#
# Track 2 (L2) — decode-phase profile with Nsight Systems (nsys).
#
# Answers "where does per-token decode time go?" for the DENSE model (Qwen2.5-14B), with
# three captures (one config each, so every timeline/summary is attributable):
#   1. depth 0,    graphs on  — baseline decode kernel mix
#   2. depth 8192, graphs on  — attention/KV cost that grows with KV depth (the L1 signal)
#   3. depth 8192, graphs off — kernel-launch overhead that CUDA graphs remove (+ residual)
#
# Each capture writes a .nsys-rep (large; open in the Nsight GUI) AND a .summary.txt (small;
# kernel-time ranking + CUDA API/launch time + memcpy time) — bring back the .summary.txt.
#
# Pinned: xeon-cg2 GPU 0. Build first with decode-temp/build-decode.sh 80.
# nsys ships with the CUDA toolkit; if it's not on PATH try `ml nsight-systems`.
#
# Usage:  sbatch decode-temp/decode-profile.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-bench"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
GPU_IDX=0
NGEN=128             # decode steps to profile (plenty for the CUDA graph to warm up + steady state)
OUTDIR="decode-profile-${SLURM_JOB_ID:-manual}"
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

# name  depth  graphs(on|off)
run_profile() {
    local name="$1" depth="$2" graphs="$3"
    local rep="$OUTDIR/$name"
    local pre=()
    [ "$graphs" = "off" ] && pre=(env GGML_CUDA_DISABLE_GRAPHS=1)   # disable CUDA graphs for this run only
    echo ">>> $name (depth=$depth graphs=$graphs)" >&2
    "${pre[@]}" nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --force-overwrite true \
        -o "$rep" \
        "$BIN" -m "$DENSE" -ngl 99 -p 0 -n "$NGEN" -d "$depth" -fa 1 -r 1 --no-warmup -o md
    # readable summaries (kernel/API/memcpy tables) so we don't need the GUI to read the result
    nsys stats --force-export=true "$rep.nsys-rep" > "$rep.summary.txt" 2>&1
    echo "    -> $rep.nsys-rep + $rep.summary.txt" >&2
}

run_profile dense_d0_graphs_on     0    on
run_profile dense_d8192_graphs_on  8192 on
run_profile dense_d8192_graphs_off 8192 off

echo "done -> $OUTDIR/  (paste the *.summary.txt back; copy the dir to ../cluster-reports/)" >&2
