#!/usr/bin/env bash
#SBATCH --job-name=dp-split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=dp-split-%j.log
#
# S18 (model-too-big): verify the data-parallel orchestrator's multi-GPU split path on models that
# do NOT fit on a single 80 GB H100. For each model, three things:
#
#   (1) GRACEFUL-FAIL  : place the too-big model on single GPUs (--dp-devices 0,1,2,3). Each replica
#                        OOMs on its card -> make_pool's all-or-nothing teardown returns nullptr ->
#                        the tool prints a clean error and exits non-zero. We assert NO crash
#                        (no segfault / abort / "core dumped"): failure isolation, not a process kill.
#   (2) SPLIT, 1 rep   : one replica spanning 2 GPUs (--dp-devices 0+1) loads + serves the model.
#                        This is the minimal capacity escape hatch.
#   (3) SPLIT, 2 reps  : two replicas, each spanning 2 GPUs (--dp-devices 0+1,2+3) -> capacity AND
#                        throughput. Swept across --dp-split layer vs row (the two split strategies).
#
# Split modes (this fork's --dp-split, same vocabulary as -sm):
#   layer  - split layers + KV across the group (pipelined; lighter interconnect)
#   row    - split layers + KV, tensor parallelism if supported (parallelized; the usual "tensor split")
#   tensor - split weights + KV (parallelized, EXPERIMENTAL) -- add to SPLITS below to include it
#
# MODELS UNDER TEST (both too big for one 80 GB card, both fit a 2x80=160 GB pair):
#   - Llama-3.3-70B-Instruct  F16   ~141 GB
#   - Mistral-Large-2407 123B Q8_0  ~130 GB
# Download (new hf CLI; both repos are gated - accept the license + `hf auth login` first):
#   hf download bartowski/Llama-3.3-70B-Instruct-GGUF \
#       --include "Llama-3.3-70B-Instruct-f16/*" --local-dir ~/models
#   hf download bartowski/Mistral-Large-Instruct-2407-GGUF \
#       --include "Mistral-Large-Instruct-2407-Q8_0/*" --local-dir ~/models
# Each lands as 4 shards in a subfolder; point -m at the FIRST shard (…-00001-of-00004.gguf) and
# llama.cpp auto-loads the rest.
#
# Cluster notes: NO GRES here - select GPUs with CUDA_VISIBLE_DEVICES, do NOT add "#SBATCH --gres".
# Modules via lmod. batched-bench uses random tokens, so no input dataset is needed.
#
# Usage: build the fork (-DGGML_CUDA=ON), download the models above, then:
#   sbatch tools/orchestrator/slurm/dp-split.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-batched-bench"

# first shard of each too-big model (the loader pulls the remaining shards automatically)
MODELS=(
  "$HOME/models/Llama-3.3-70B-Instruct-f16/Llama-3.3-70B-Instruct-f16-00001-of-00004.gguf"
  "$HOME/models/Mistral-Large-Instruct-2407-Q8_0/Mistral-Large-Instruct-2407-Q8_0-00001-of-00004.gguf"
)

CTX=4096             # per-replica context; kept conservative so the 141 GB F16 + KV fits a 2-GPU
                     # pair (~19 GB headroom over 160 GB). The 130 GB Q8 model has more room to raise it.
NGL=99               # offload all layers to the GPU(s) (orchestrator warns, does not force this)
NBATCH=2048          # logical batch (-b)
UBATCH=512           # physical micro-batch (-ub)
PP=512               # prompt length (single value keeps the capacity demo short)
TG=128               # generation length
PL=1,2,4             # parallel sequences per replica (small: this is a capacity test, not a sweep)

SPLITS=(layer row tensor)   # split strategies to compare; add 'tensor' to also test the experimental mode
# -----------------------------------------------------------------------------

ml gcc cuda          # pin versions if needed: ml gcc/15.2 cuda/13.1

RESULTS="results/dp-split-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# run a labelled command; capture full output, echo the table rows. returns the command's exit code.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    grep -E "^\|" "$log" || echo "  (exit 0 but no table rows - see $log)"
  else
    echo "  exit $rc - tail:"; tail -n 12 "$log" | sed 's/^/    /'
  fi
  echo
  return $rc
}

for MODEL in "${MODELS[@]}"; do
  name="$(basename "$MODEL" .gguf)"
  echo "##################################################################"
  echo "# model: $name"
  echo "#   $MODEL"
  echo "##################################################################"
  if [ ! -f "$MODEL" ]; then
    echo "  !! not found - did the hf download finish? skipping."; echo; continue
  fi

  # --- (1) graceful-fail: too-big model onto single GPUs; assert a CLEAN failure (no crash) ---
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  GF_LOG="$RESULTS/${name}.graceful-fail.log"
  run "GRACEFUL-FAIL | $name | --dp-devices 0,1,2,3 (expect clean non-zero exit)" \
      "$GF_LOG" \
      "$BIN" -m "$MODEL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
             -npp "$PP" -ntg "$TG" -npl "$PL" --dp-devices 0,1,2,3
  gf_rc=$?
  if grep -qiE "segmentation|core dumped|terminate called|signal|Aborted|CUDA error: out of memory.*abort" "$GF_LOG"; then
    echo "  >> GRACEFUL-FAIL: *** CRASHED *** (not a clean teardown) - investigate $GF_LOG"
  elif [ $gf_rc -ne 0 ] && grep -qiE "failed to build the data-parallel replica pool|failed to LOAD MODEL" "$GF_LOG"; then
    echo "  >> GRACEFUL-FAIL: PASS - non-zero exit with a clean error, no crash (failure isolation works)"
  elif [ $gf_rc -eq 0 ]; then
    echo "  >> GRACEFUL-FAIL: model LOADED on a single card - it is NOT too big; pick a larger model/quant"
  else
    echo "  >> GRACEFUL-FAIL: non-zero exit but no recognized clean-error line - check $GF_LOG"
  fi
  echo

  # --- (2)+(3) split serve: one then two replicas spanning 2 GPUs, swept over split mode ---
  for SM in "${SPLITS[@]}"; do
    # (2) single split replica over GPUs 0+1 (uses 2 cards)
    export CUDA_VISIBLE_DEVICES=0,1
    run "SPLIT 1-rep | $name | 0+1 | --dp-split $SM" \
        "$RESULTS/${name}.split1.${SM}.log" \
        "$BIN" -m "$MODEL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
               -npp "$PP" -ntg "$TG" -npl "$PL" --dp-devices 0+1 --dp-split "$SM"

    # (3) two split replicas over 0+1 and 2+3 (uses all 4 cards): capacity + throughput
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    run "SPLIT 2-rep | $name | 0+1,2+3 | --dp-split $SM" \
        "$RESULTS/${name}.split2.${SM}.log" \
        "$BIN" -m "$MODEL" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
               -npp "$PP" -ntg "$TG" -npl "$PL" --dp-devices 0+1,2+3 --dp-split "$SM"
  done
done

echo "done -> $RESULTS"
echo "interpretation:"
echo "  - GRACEFUL-FAIL must say PASS (clean non-zero exit, no crash). If it LOADED, the model fits"
echo "    one card - use a larger model/quant (see MODELS UNDER TEST in this script's header)."
echo "  - SPLIT 1-rep proves one replica spanning 2 GPUs loads + serves the too-big model."
echo "  - SPLIT 2-rep 'ALL' row is the aggregate over two split replicas; compare layer vs row for"
echo "    the throughput/interconnect trade-off. row uses tensor parallelism, layer is pipelined."
