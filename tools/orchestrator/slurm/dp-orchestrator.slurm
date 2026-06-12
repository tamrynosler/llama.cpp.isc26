#!/usr/bin/env bash
#SBATCH --job-name=ppl-dp-orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=ppl-dp-orch-%j.log
#
# Fork (llama.cpp.isc26) perplexity throughput across replicas. For each model:
#   (A) 1 GPU,  default batching   -> fork single-GPU baseline (PPL + throughput)
#   (B) 4 GPUs, --data-parallel 4  -> orchestrator scaling     (PPL + throughput)
#
# With n_seq stacking, the DP run uses the same per-decode batching as the 1-GPU run
# (n_seq = batch_size / n_ctx), so run (B) PPL must equal run (A) PPL EXACTLY. Throughput
# is the end-to-end WALL / tok/s line, plus the DP "compute-only" time (excludes model
# load). NOTE: the orchestrator loads its replicas sequentially, so the DP end-to-end WALL
# includes ~N x model load - use the compute-only line for the pure scaling number.
#
# Cluster notes: NO GRES on this cluster - select GPUs with CUDA_VISIBLE_DEVICES,
# do NOT add "#SBATCH --gres". Modules via lmod.
#
# Usage: fill in the variables, build the fork (-DGGML_CUDA=ON), then:
#   sbatch tools/orchestrator/slurm/dp-orchestrator.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-perplexity"
INPUT="$HOME/datasets/wikitext-2-raw/wiki.test.raw"
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"        # SAME 5 models as baseline-original.slurm
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
  "$HOME/models/OLMo-2-1124-13B-Instruct-Q4_0.gguf"
  "$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)
CTX=512        # perplexity context window
NGL=99         # offload all layers to the GPU (required: orchestrator warns, does not force this)
DP=4           # replicas across the node's 4 GPUs
# -----------------------------------------------------------------------------

ml gcc cuda    # pin versions if needed: ml gcc/15.2 cuda/13.1

TIME=""; command -v /usr/bin/time >/dev/null 2>&1 && TIME="/usr/bin/time -v"

RESULTS="results/dp-orch-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# run <label> <logfile> <command...> - times the run end-to-end, then extracts PPL and
# computes throughput (tokens = chunks * n_ctx, both parsed from the tool's own log line).
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  local t0 t1 dt
  t0=$(date +%s.%N)
  if $TIME "$@" > "$log" 2>&1; then
    t1=$(date +%s.%N)
    dt=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}")
    local chunks nctx toks tps ppl compute memkb
    chunks=$(grep -oE "over [0-9]+ chunks" "$log" | grep -oE "[0-9]+" | head -1)
    nctx=$(grep -oE "n_ctx=[0-9]+"         "$log" | grep -oE "[0-9]+" | head -1)
    ppl=$(grep "Final estimate" "$log" | tail -1 | sed -E 's/^[0-9. ]*I //')
    compute=$(grep -oE "[0-9]+ chunks across [0-9]+ replicas in [0-9.]+ s" "$log" | tail -1)
    memkb=$(grep -oE "Maximum resident set size \(kbytes\): [0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
    echo "  ${ppl:-(no Final estimate)}"
    if [ -n "$chunks" ] && [ -n "$nctx" ]; then
      toks=$(( chunks * nctx ))
      tps=$(awk "BEGIN{printf \"%.1f\", $toks/($t1-$t0)}")
      echo "  WALL=${dt}s  TOKENS=${toks}  THROUGHPUT=${tps} tok/s (end-to-end, incl. load)"
    else
      echo "  WALL=${dt}s  (could not parse chunks/n_ctx for tok/s)"
    fi
    [ -n "$compute" ] && echo "  DP compute-only: $compute"
    [ -n "$memkb" ]   && echo "  peak RSS: $(awk "BEGIN{printf \"%.1f\", $memkb/1024}") MB"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"

  # (A) 1 GPU, default batching - fork single-GPU baseline + reference PPL
  export CUDA_VISIBLE_DEVICES=0
  run "FORK | $name | 1 GPU | default" \
      "$RESULTS/${name}.1gpu-default.log" \
      "$BIN" -m "$M" -bf "$INPUT" --ctx-size "$CTX" -ngl "$NGL" -b 512

  # (B) 4 GPUs via the orchestrator - throughput scaling + PPL (must equal run A)
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  run "FORK | $name | 4 GPUs | --data-parallel $DP" \
      "$RESULTS/${name}.4gpu-dp.log" \
      "$BIN" -m "$M" -bf "$INPUT" --ctx-size "$CTX" -ngl "$NGL" -b 512 \
             --data-parallel "$DP" --dp-devices 0,1,2,3
done

echo "done -> $RESULTS"
echo "legality:   per model, 'Final estimate' PPL of *.4gpu-dp.log must equal *.1gpu-default.log"
echo "throughput: compare THROUGHPUT (and DP compute-only) of *.4gpu-dp.log vs *.1gpu-default.log"
