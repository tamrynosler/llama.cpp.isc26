#!/usr/bin/env bash
#SBATCH --job-name=ppl-baseline-orig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=ppl-baseline-orig-%j.log
#
# Baseline perplexity on the ORIGINAL (upstream) llama.cpp checkout, 5 models, ONE GPU.
# This is the reference PPL + throughput the fork's single-replica path is compared against
# (does our fork regress vs upstream?).
#
# Cluster notes: this cluster has NO GRES - every node has 4x H100 and we select GPUs with
# CUDA_VISIBLE_DEVICES. Do NOT add "#SBATCH --gres". Modules via lmod.
#
# Usage: fill in the variables below, build the original repo first, then:
#   sbatch tools/orchestrator/slurm/baseline-original.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
ORIG_REPO="$HOME/llama.cpp"                              # original upstream checkout
ORIG_BIN="$ORIG_REPO/build/bin/llama-perplexity"        # build with -DGGML_CUDA=ON first
INPUT="$HOME/datasets/wikitext-2-raw/wiki.test.raw"     # perplexity input text (see README)
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"        # SAME 5 models as dp-orchestrator.slurm
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
  "$HOME/models/OLMo-2-1124-13B-Instruct-Q4_0.gguf"
  "$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)
CTX=512        # standard perplexity context window
NGL=99         # offload all layers to the GPU
# -----------------------------------------------------------------------------

ml gcc cuda    # pin versions if your site needs them, e.g.: ml gcc/15.2 cuda/13.1

export CUDA_VISIBLE_DEVICES=0   # single-GPU baseline

TIME=""; command -v /usr/bin/time >/dev/null 2>&1 && TIME="/usr/bin/time -v"

RESULTS="results/baseline-orig-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L 2>/dev/null || true
( cd "$ORIG_REPO" && echo "orig repo commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
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
    local chunks nctx toks tps ppl memkb
    chunks=$(grep -oE "over [0-9]+ chunks" "$log" | grep -oE "[0-9]+" | head -1)
    nctx=$(grep -oE "n_ctx=[0-9]+"         "$log" | grep -oE "[0-9]+" | head -1)
    ppl=$(grep "Final estimate" "$log" | tail -1 | sed -E 's/^[0-9. ]*I //')
    memkb=$(grep -oE "Maximum resident set size \(kbytes\): [0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
    echo "  ${ppl:-(no Final estimate)}"
    if [ -n "$chunks" ] && [ -n "$nctx" ]; then
      toks=$(( chunks * nctx ))
      tps=$(awk "BEGIN{printf \"%.1f\", $toks/($t1-$t0)}")
      echo "  WALL=${dt}s  TOKENS=${toks}  THROUGHPUT=${tps} tok/s (end-to-end, incl. load)"
    else
      echo "  WALL=${dt}s  (could not parse chunks/n_ctx for tok/s)"
    fi
    [ -n "$memkb" ] && echo "  peak RSS: $(awk "BEGIN{printf \"%.1f\", $memkb/1024}") MB"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"
  run "ORIGINAL | $name | 1 GPU | default" \
      "$RESULTS/${name}.1gpu.log" \
      "$ORIG_BIN" -m "$M" -bf "$INPUT" --ctx-size "$CTX" -ngl "$NGL"
done

echo "done -> $RESULTS"
