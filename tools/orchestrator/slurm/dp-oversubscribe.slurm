#!/usr/bin/env bash
#SBATCH --job-name=dp-oversub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=dp-oversub-%j.log
#
# S20 (model-too-small): multiple replicas PER GPU (oversubscription). A small model underutilizes
# an H100, so co-resident replicas (each its own context/stream) overlap and lift aggregate
# throughput. This sweep finds the per-GPU sweet spot and documents the memory cost.
#
# Each replica is a FULL, independent weight copy (data-parallel) - so R replicas on a card cost
# R x (weights + KV). For a small model that is cheap; the sweep stops helping once the GPU's
# compute saturates (or VRAM runs out). Done when an R>1 config beats R=1 (4-on-4) for a small model.
#
# Placement uses the count-based flags:
#   --dp-num-devices D --dp-replicas-per-device R   ->  R replicas on each of GPUs 0..D-1 (D*R total)
# (equivalent to the explicit --dp-devices 0,0,..,1,1,.. list; counts are just terser.)
#
# Cluster notes: NO GRES here - select GPUs with CUDA_VISIBLE_DEVICES, do NOT add "#SBATCH --gres".
# Modules via lmod. batched-bench uses random tokens, so no input dataset is needed.
#
# Usage: build the fork (-DGGML_CUDA=ON), then:
#   sbatch tools/orchestrator/slurm/dp-oversubscribe.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-batched-bench"

# small models with memory to spare (best oversubscription candidates: most GPU idle to fill)
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
)

NUM_DEVICES=4                 # use all 4 GPUs on the node
REPS_PER_DEVICE=(1 2 3 4)     # sweep replicas/GPU: 1=baseline (4-on-4), 2=8-on-4, 3=12-on-4, 4=16-on-4
CTX=8192                      # per-replica context (small models -> cheap KV even x R)
NGL=99                        # all layers on GPU
NBATCH=2048
UBATCH=512
PP=128                        # short prompt/gen: this tests concurrency, not long sequences
TG=64
PL=1,4,16                     # parallel sequences PER replica (low pl = more GPU idle for stacking)
# -----------------------------------------------------------------------------

ml gcc cuda

export CUDA_VISIBLE_DEVICES=0,1,2,3

RESULTS="results/dp-oversub-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# run one config with a background per-GPU memory sampler; echo the ALL aggregate rows + peak VRAM.
run_cfg() {
  local label="$1"; local log="$2"; local memlog="$3"; shift 3
  echo "=== $label ==="
  # sample memory.used every 0.5 s while the bench runs (peak is just after all replicas load)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 > "$memlog" 2>/dev/null &
  local smpl=$!
  "$@" > "$log" 2>&1
  local rc=$?
  kill "$smpl" 2>/dev/null; wait "$smpl" 2>/dev/null
  if [ $rc -eq 0 ]; then
    grep -E "^\| *ALL" "$log" || echo "  (exit 0 but no ALL rows - see $log)"
    # peak memory.used per GPU across the run
    awk -F', *' 'NF==2 {if ($2+0 > m[$1]+0) m[$1]=$2} END {printf "  peak VRAM:"; for (g=0; g<4; g++) printf " GPU%d=%sMiB", g, m[g]; print ""}' "$memlog"
  else
    echo "  exit $rc - tail:"; tail -n 12 "$log" | sed 's/^/    /'
  fi
  echo
}

for MODEL in "${MODELS[@]}"; do
  name="$(basename "$MODEL" .gguf)"
  echo "##################################################################"
  echo "# model: $name   (devices=$NUM_DEVICES)"
  echo "##################################################################"
  if [ ! -f "$MODEL" ]; then
    echo "  !! not found - skipping."; echo; continue
  fi

  for R in "${REPS_PER_DEVICE[@]}"; do
    total=$(( NUM_DEVICES * R ))
    run_cfg "OVERSUB | $name | ${R}/GPU = ${total} replicas on ${NUM_DEVICES} GPUs" \
            "$RESULTS/${name}.r${R}.log" \
            "$RESULTS/${name}.r${R}.mem" \
            "$BIN" -m "$MODEL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
                   -npp "$PP" -ntg "$TG" -npl "$PL" \
                   --dp-num-devices "$NUM_DEVICES" --dp-replicas-per-device "$R"
  done
done

echo "done -> $RESULTS"
echo "interpretation:"
echo "  - For each model, compare the ALL-row 'S t/s' across R=1,2,3,4 at a fixed pl. The sweet spot"
echo "    is the largest R where aggregate throughput still climbs; beyond it the GPU is compute-"
echo "    saturated (no gain) or VRAM-bound. Oversubscription is proven if any R>1 beats R=1."
echo "  - 'peak VRAM' per GPU shows the memory budget: ~R x (weights + KV). Confirm it stays under 80 GB."
echo "  - Low-pl rows show the effect most: a single replica leaves the GPU idle, so stacking helps;"
echo "    at high pl a single replica may already saturate, shrinking the oversubscription win."
