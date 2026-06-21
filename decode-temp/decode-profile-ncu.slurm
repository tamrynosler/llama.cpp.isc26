#!/usr/bin/env bash
#SBATCH --job-name=decode-ncu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --nodelist=xeon-cg2
#SBATCH --output=decode-ncu-%j.log
#
# Track 2 (L2b) — per-kernel Nsight Compute drill on the decode kernels nsys flagged.
#
# nsys (decode-profile.slurm) said decode is GPU-bound on the quantized GEMV (mul_mat_vec_q,
# ~73% of decode), with quantize_q8_1 + rms_norm next. This run answers the deciding question:
# are those kernels BANDWIDTH-SATURATED (-> the only win is moving fewer bytes) or do they have
# throughput/occupancy headroom (-> optimize the kernel)?  The SpeedOfLight section reports
# Memory Throughput [% of peak] and Compute (SM) Throughput [% of peak]; MemoryWorkloadAnalysis
# reports achieved DRAM GB/s.
#
# Runs on the DENSE model, pure decode (llama-simple), CUDA graphs disabled so ncu's
# launch-skip/launch-count land deterministically on steady-state decode kernels (the per-kernel
# metrics are identical whether the kernel is launched directly or via a graph).
#
# Pinned: xeon-cg2 GPU 0. Build first with decode-temp/build-decode.sh 80.
# ncu ships with the CUDA toolkit; if it's not on PATH try `ml nsight-compute`.
#
# REQUIRES GPU performance-counter access. If you see "ERR_NVGPUCTRPERM", the node restricts
# counters to admins — ask the cluster admin to set, in /etc/modprobe.d/:
#     options nvidia NVreg_RestrictProfilingToAdminUsers=0
# and reload the nvidia module / reboot. (nsys does NOT need this; only ncu does.)
#
# Usage:  sbatch decode-temp/decode-profile-ncu.slurm

set -uo pipefail

# ---- config -----------------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-simple"
DENSE="/shared/llms/models/Qwen2.5-14B-Instruct-Q4_0.gguf"
GPU_IDX=0
NGEN=64
PROMPT="The history of computing began"
KERNELS="mul_mat_vec_q|quantize_q8_1|rms_norm_f32|flash_attn_ext_f16"
SKIP=300             # skip prompt-prefill + early decode kernels -> land in steady state
COUNT=32             # profile this many matching kernel launches (ncu replays each; keep modest)
OUTDIR="decode-ncu-${SLURM_JOB_ID:-manual}"
REP="$OUTDIR/dense_decode"
# -----------------------------------------------------------------------------

ml gcc cuda
export CUDA_VISIBLE_DEVICES="$GPU_IDX"

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not on PATH. Try:  ml nsight-compute  (or the site's Nsight Compute module)" >&2
    exit 1
fi

mkdir -p "$OUTDIR"
echo "node=$(hostname) commit=$(git -C "$REPO" rev-parse --short HEAD)" >&2
ncu --version | head -1 >&2

# --clock-control none: don't try to lock GPU clocks (needs privileges on many clusters); metrics
# are still valid. GGML_CUDA_DISABLE_GRAPHS=1: deterministic launch ordering for skip/count.
GGML_CUDA_DISABLE_GRAPHS=1 ncu \
    --clock-control none \
    --kernel-name "regex:$KERNELS" \
    --launch-skip "$SKIP" --launch-count "$COUNT" \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -f -o "$REP" \
    "$BIN" -m "$DENSE" -ngl 99 -n "$NGEN" "$PROMPT"

# human-readable export (no GUI needed): per-kernel SpeedOfLight + memory tables
ncu --import "$REP.ncu-rep" --page details > "$REP.details.txt" 2>&1 || \
    echo "WARN: details export failed; open $REP.ncu-rep in the Nsight Compute GUI" >&2

echo "done -> $OUTDIR/  (paste $REP.details.txt back; copy the dir to ../cluster-reports/)" >&2
