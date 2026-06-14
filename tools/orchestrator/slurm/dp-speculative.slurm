#!/usr/bin/env bash
#SBATCH --job-name=dp-spec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cn002
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=dp-spec-%j.log
#
# Data-parallel SPECULATIVE decoding: combine the speculative-decoding speedup with the orchestrator's
# replica speedup. For each (target, draft) pair and draft length, compare:
#   (A) 1 GPU,  single spec stream (no -dp)        -> baseline decode tok/s + acceptance%
#   (B) 4 GPUs, -dp 4 (one target+draft per GPU)   -> ALL-row aggregate decode tok/s (weak scaling)
# Each of the 4 replicas runs the byte-identical single-stream spec decode on the SAME prompt
# concurrently, so the aggregate decode t/s should approach 4x the 1-GPU number with acceptance% and
# n_accept/replica UNCHANGED (the spec speedup is preserved, then multiplied by the replica count).
#
# Single-node DP needs NO MPI (the orchestrator lib is always built). Build the fork once:
#   ml gcc cuda
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#
# Cluster notes: NO GRES - select GPUs with CUDA_VISIBLE_DEVICES, no "#SBATCH --gres". cn002 is in the
# default partition, so no --partition needed. Modules via lmod.
#
# Usage: fill in the variables, build (above), then:
#   sbatch tools/orchestrator/slurm/dp-speculative.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build/bin/llama-speculative"
MODELS_DIR=/shared/llms/models
DATASET=/shared/llms/datasets/wikitext-2-raw/wiki.test.raw

# target|draft|label  (Qwen2 tokenizer family - vocab must match between target and draft)
SPEC_PAIRS=(
  "Qwen2.5-72B-Instruct-Q4_K_M.gguf|qwen2.5-0.5b-instruct-q4_k_m.gguf|72B+0.5B"
  "Qwen2.5-72B-Instruct-Q4_K_M.gguf|Qwen2.5-7B-Instruct-Q4_K_M.gguf|72B+7B"
)
DMAX_SWEEP=(4 8 16)   # --spec-draft-n-max values to sweep (tokens drafted per step); 0 = no draft, skipped
DMIN=2                # --spec-draft-n-min
N_PREDICT=256         # tokens to generate per stream
PROMPT_CHARS=4096     # bytes sliced from the dataset for the prompt (same prompt on every replica)
DP=4                  # replicas (one target+draft pair per GPU)

CTX=8192
BATCH=2048
UBATCH=512
NGL=99                # target GPU layers
NGLD=99               # draft GPU layers (-ngld)
CTK=f16
CTV=f16
# -----------------------------------------------------------------------------

ml gcc cuda           # pin if needed: ml gcc/15.2 cuda/13.3

RESULTS="$REPO/results/dp-spec-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

# slice a prompt from the dataset (no separate file needed); same prompt feeds every replica.
PROMPT_FILE="$(mktemp /tmp/dp_spec_prompt_XXXX.txt)"
head -c "$PROMPT_CHARS" "$DATASET" > "$PROMPT_FILE"
trap 'rm -f "$PROMPT_FILE"' EXIT

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "prompt: first ${PROMPT_CHARS} bytes of $(basename "$DATASET")"
echo

# run <label> <logfile> <command...> - capture full output, echo the lines that carry the numbers.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then
    # single-stream prints "decoded ... t/s" + "accept ="; the DP path prints the table + aggregate.
    grep -E "decoded .* t/s|accept +=|^\| *(REP|[0-9]|ALL)|aggregate decode" "$log" || echo "  (no result lines - see $log)"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

for pair in "${SPEC_PAIRS[@]}"; do
  IFS='|' read -r target draft plabel <<< "$pair"
  echo "######## pair: $plabel  (target=$target  draft=$draft) ########"

  for dmax in "${DMAX_SWEEP[@]}"; do
    common=( -m "$MODELS_DIR/$target" -md "$MODELS_DIR/$draft"
             -c "$CTX" -b "$BATCH" -ub "$UBATCH" -ngl "$NGL" -ngld "$NGLD"
             -ctk "$CTK" -ctv "$CTV"
             --spec-draft-n-max "$dmax" --spec-draft-n-min "$DMIN"
             -n "$N_PREDICT" -f "$PROMPT_FILE" )

    # (A) 1 GPU, single spec stream (baseline)
    export CUDA_VISIBLE_DEVICES=0
    run "SPEC | $plabel | dmax=$dmax | 1 GPU baseline" \
        "$RESULTS/${plabel}.dmax${dmax}.1gpu.log" \
        "$BIN" "${common[@]}"

    # (B) 4 GPUs, -dp 4 (one target+draft per GPU) - weak-scaling aggregate
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    run "SPEC | $plabel | dmax=$dmax | ${DP} GPUs --data-parallel" \
        "$RESULTS/${plabel}.dmax${dmax}.${DP}gpu-dp.log" \
        "$BIN" "${common[@]}" --data-parallel "$DP" --dp-devices 0,1,2,3
  done
done

echo "done -> $RESULTS"
echo "verify: per (pair,dmax), the DP 'ALL' row decode t/s should be ~${DP}x the 1-GPU 'decoded ... t/s',"
echo "        with each replica's n_accept and accept% matching the 1-GPU run (spec speedup preserved,"
echo "        then multiplied by replica count). acceptance% is set by the model pair + dmax, not by DP."
