#!/usr/bin/env bash
#SBATCH --job-name=dp-spec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cn002
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=dp-spec-%j.log
#
# Data-parallel SPECULATIVE decoding, STRONG scaling (the real-work metric): a corpus is sliced into M
# DISTINCT prompts and processed across the replicas - NOT the same prompt duplicated. For each
# (target, draft) pair + draft length, compare:
#   (A) 1 GPU,  -dp 1 --dp-chunk-chars C  -> serial baseline: all M prompts on one GPU (wall = T1)
#   (B) 4 GPUs, -dp 4 --dp-chunk-chars C  -> work-stealing across 4 (target+draft) pairs (wall = T4)
# REAL speedup = T1 / T4 (how much faster 4 GPUs chew through the SAME distinct workload). This is the
# offline-batch story: split the dataset, process the shards in parallel. The spec speedup is inside
# each stream; DP multiplies the batch throughput.
#
# (For the identical-stream scaling-efficiency diagnostic, add --dp-weak instead - see RESULTS.md
# 2026-06-14; it is NOT a real-work throughput number and is not run here.)
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
DMAX_SWEEP=(4 8 16)   # --spec-draft-n-max values to sweep (tokens drafted per step)
DMIN=2                # --spec-draft-n-min
N_PREDICT=256         # tokens generated per prompt
DP=4                  # replicas (one target+draft pair per GPU)
SEED=42               # fixed seed (reproducible; distinct prompts still diverge - that's the point)

CHUNK_CHARS=4096      # bytes per shard -> one distinct prompt each (~the old single-prompt size)
NUM_CHUNKS=16         # M distinct prompts sliced from the dataset (16 across 4 GPUs = 4 each)
CORPUS_CHARS=$(( CHUNK_CHARS * NUM_CHUNKS ))

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

# slice a CORPUS from the dataset; the tool slices it into NUM_CHUNKS distinct prompts of CHUNK_CHARS.
CORPUS_FILE="$(mktemp /tmp/dp_spec_corpus_XXXX.txt)"
head -c "$CORPUS_CHARS" "$DATASET" > "$CORPUS_FILE"
trap 'rm -f "$CORPUS_FILE"' EXIT

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "corpus: first ${CORPUS_CHARS} bytes of $(basename "$DATASET") -> ${NUM_CHUNKS} x ${CHUNK_CHARS}-byte prompts"
echo

# run <label> <logfile> <command...> - capture full output, echo the strong-scaling result lines.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then
    grep -E "strong scaling:|prompts ok|gen tokens in|load balance|^\| *[0-9]" "$log" \
      || echo "  (no result lines - see $log)"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

# pull the wall-clock seconds out of a strong-run log ("... in 12.185s wall ...").
wall_of() { sed -nE 's/.* in ([0-9.]+)s wall.*/\1/p' "$1" | head -1; }

for pair in "${SPEC_PAIRS[@]}"; do
  IFS='|' read -r target draft plabel <<< "$pair"
  echo "######## pair: $plabel  (target=$target  draft=$draft) ########"

  for dmax in "${DMAX_SWEEP[@]}"; do
    common=( -m "$MODELS_DIR/$target" -md "$MODELS_DIR/$draft"
             -c "$CTX" -b "$BATCH" -ub "$UBATCH" -ngl "$NGL" -ngld "$NGLD"
             -ctk "$CTK" -ctv "$CTV" -s "$SEED"
             --spec-draft-n-max "$dmax" --spec-draft-n-min "$DMIN"
             -n "$N_PREDICT" -f "$CORPUS_FILE" --dp-chunk-chars "$CHUNK_CHARS" )

    base_log="$RESULTS/${plabel}.dmax${dmax}.1gpu-serial.log"
    par_log="$RESULTS/${plabel}.dmax${dmax}.${DP}gpu-strong.log"

    # (A) 1 GPU serial baseline: all M prompts on one GPU
    export CUDA_VISIBLE_DEVICES=0
    run "SPEC strong | $plabel | dmax=$dmax | 1 GPU serial baseline" \
        "$base_log" \
        "$BIN" "${common[@]}" --data-parallel 1

    # (B) DP GPUs strong: M prompts work-stolen across DP (target+draft) pairs
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    run "SPEC strong | $plabel | dmax=$dmax | ${DP} GPUs work-stealing" \
        "$par_log" \
        "$BIN" "${common[@]}" --data-parallel "$DP" --dp-devices 0,1,2,3

    # real speedup = serial wall / parallel wall
    t1="$(wall_of "$base_log")"; t4="$(wall_of "$par_log")"
    awk -v a="$t1" -v b="$t4" -v dp="$DP" 'BEGIN{
      if (a != "" && b != "" && b+0 > 0)
        printf "  >> REAL STRONG SPEEDUP (1 GPU / %d GPU): %ss / %ss = %.2fx\n\n", dp, a, b, a/b;
      else print "  >> speedup: could not parse walls (see logs)\n";
    }'
  done
done

echo "done -> $RESULTS"
echo "headline: per (pair,dmax) the 'REAL STRONG SPEEDUP' = serial-wall / parallel-wall over the SAME"
echo "          M distinct prompts (ideal = ${DP}). per-replica 'jobs' columns show the work balance;"
echo "          'load balance' = sum(per-prompt time)/wall (~${DP} when balanced). 'aggregate decode t/s'"
echo "          is generated tokens / wall on distinct work - the real offline-batch throughput."
