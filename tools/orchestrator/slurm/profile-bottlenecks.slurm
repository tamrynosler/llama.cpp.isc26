#!/usr/bin/env bash
#SBATCH --job-name=profile-bottlenecks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=profile-bottlenecks-%j.log
#
# Profile WHERE the time actually goes before optimising the bench/perplexity
# harnesses. This answers, with measurements, what the optimisation docs assert.
#
# TWO models on purpose - they stress opposite things:
#   MODEL      large-VOCAB, single-card (Gemma-2-27B, 256k vocab). The perplexity
#              CPU post-proc (log_softmax/expf over the vocab) scales with VOCAB,
#              not parameter count - a big small-vocab model would HIDE it. Kept on
#              one GPU so the GPU-vs-CPU split is clean (no cross-GPU comms).
#   MODEL_BIG  large-PARAM, multi-GPU (Llama-3.3-70B Q8). Used only where "large"
#              is the point: the pipeline-parallel gate + multi-GPU throughput.
#
# Measurements:
#   M1  Is perplexity CPU-bound? -> decode-vs-postproc split, n_seq=1 vs n_seq=8.
#       GATE for the perplexity host-side tasks (OpenMP P2, omp-simd exp P4,
#       double-buffer P5). Full corpus (the cheap, accurate number). If CPU
#       post-proc is single-digit %%, do NOT spend effort on P2/P4/P5.
#       Uses the LLAMA_PPL_PROFILE timer in perplexity.cpp (env-gated; cannot
#       change the PPL).
#   M2  Is pipeline parallelism ON across GPUs? (the single biggest multi-GPU
#       throughput switch). Runs MODEL_BIG on 2 GPUs and contrasts -ngl 999
#       (gate passes) vs -ngl = n_layer (gate fails - the silent footgun).
#   M3  CPU hot spots -> validates the mechanism of P2/P3/P4: per-chunk thread
#       spawn (clone count), mutex contention (futex), scalar expf over vocab.
#   M4  GPU timeline -> P5 (does the GPU idle during NLL post-proc?) and B1 (does
#       the first batched-bench cell pay a one-time warmup cost vs its neighbours?).
#   M5  Would P4 even help? -> checks the built binary actually links a vectorised
#       expf (libmvec _ZGV*). `#pragma omp simd` over scalar libm expf does NOT
#       vectorise without it; if absent, P4 is a no-op.
#
# perf/nsys/strace are each optional - missing ones are skipped, not fatal.
#
# Usage: build the fork (-DGGML_CUDA=ON -DGGML_NATIVE=ON), download models +
# wikitext (see the hf/dataset commands in the chat / this header), set the paths
# below, then:
#   sbatch tools/orchestrator/slurm/profile-bottlenecks.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
PPL="$REPO/build/bin/llama-perplexity"
BB="$REPO/build/bin/llama-batched-bench"

# large-VOCAB, fits one 80 GB H100 (~29 GB) - the perplexity CPU-bottleneck model
MODEL="$HOME/models/gemma-2-27b-it-Q8_0.gguf"
# large-PARAM, spans 2 GPUs (~75 GB) - the multi-GPU pipeline-parallel model.
# bartowski splits the Q8 into shards in a subfolder; point at the FIRST shard.
MODEL_BIG="$HOME/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf"

WIKI="$HOME/datasets/wikitext-2-raw/wiki.test.raw"   # run scripts/get-wikitext-2.sh from $REPO first

CTX=2048             # perplexity chunk window (-c)
NSEQ_HI=8            # batched chunks-per-decode for M1b: -b = CTX*NSEQ_HI
CHUNKS_M1=""         # "" = full corpus (the real M1 number). Set e.g. 60 for a quick loop.
CHUNKS_FAST=20       # short run for the heavy profilers (strace/nsys add big overhead)
BB_CTX=20480         # batched-bench KV budget: must cover max (npp+ntg)*max(npl)
# -----------------------------------------------------------------------------

ml gcc cuda          # pin if needed: ml gcc/15.2 cuda/13.1

RESULTS="results/profile-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "allocation: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  (node has $(nproc --all 2>/dev/null || echo '?') logical CPUs)"
echo

for f in "$PPL" "$BB" "$MODEL" "$WIKI"; do
  [ -e "$f" ] || { echo "!! missing: $f  (build the fork / set MODEL / fetch wikitext) - aborting"; exit 1; }
done

# build the optional --chunks flag for M1 (empty CHUNKS_M1 -> full corpus)
M1CHUNKS=(); [ -n "$CHUNKS_M1" ] && M1CHUNKS=(--chunks "$CHUNKS_M1")

# =============================================================================
echo "############## M1: perplexity decode-vs-postproc split (P2/P4/P5 gate) ##"
# =============================================================================
echo "--- M1a: -c $CTX (n_seq=1), full corpus ---"
LLAMA_PPL_PROFILE=1 "$PPL" -m "$MODEL" -f "$WIKI" -ngl 999 -fa on \
    -c "$CTX" "${M1CHUNKS[@]}" > "$RESULTS/m1a.log" 2>&1
grep -E "\[profile\]|Final estimate" "$RESULTS/m1a.log" || tail -n 5 "$RESULTS/m1a.log"

echo "--- M1b: -c $CTX -b $((CTX*NSEQ_HI)) (n_seq=$NSEQ_HI, batched), full corpus ---"
LLAMA_PPL_PROFILE=1 "$PPL" -m "$MODEL" -f "$WIKI" -ngl 999 -fa on \
    -c "$CTX" -b "$((CTX*NSEQ_HI))" "${M1CHUNKS[@]}" > "$RESULTS/m1b.log" 2>&1
grep -E "\[profile\]|Final estimate" "$RESULTS/m1b.log" || tail -n 5 "$RESULTS/m1b.log"
echo "  >> CPU post-proc %% small (single digits) -> skip P2/P4/P5. If 15-40%%+"
echo "     (large-vocab models) they are worth doing. n_seq=8 makes the GPU more"
echo "     efficient, so its post-proc share should be >= the n_seq=1 share -"
echo "     that growth is exactly the slack P2/P4/P5 would reclaim."
echo

# =============================================================================
echo "############## M2: pipeline parallelism on/off (multi-GPU throughput) ###"
# =============================================================================
if [ ! -e "$MODEL_BIG" ]; then
  echo "  MODEL_BIG not found ($MODEL_BIG) - skip M2 (download the 70B Q8 to run it)"
elif nvidia-smi -L 2>/dev/null | grep -q "GPU 1"; then
  export CUDA_VISIBLE_DEVICES=0,1
  echo "--- M2a: MODEL_BIG on 2 GPUs, -ngl 999 -sm layer (gate SHOULD pass) ---"
  "$BB" -m "$MODEL_BIG" -ngl 999 -sm layer -fa on -c 4096 -npp 128 -ntg 32 -npl 1 \
      > "$RESULTS/m2a.log" 2>&1
  grep -iE "pipeline parallelism enabled" "$RESULTS/m2a.log" \
      && echo "  >> PASS: pipeline parallelism ENABLED" \
      || echo "  >> NOT enabled (check $RESULTS/m2a.log)"
  NLAYER=$(grep -aoE "n_layer[[:space:]=]+[0-9]+" "$RESULTS/m2a.log" | grep -oE "[0-9]+$" | head -1)
  echo "  (MODEL_BIG n_layer detected: ${NLAYER:-unknown})"
  if [ -n "${NLAYER:-}" ]; then
    echo "--- M2b: -ngl $NLAYER (== n_layer; gate SHOULD fail - needs STRICTLY > ) ---"
    "$BB" -m "$MODEL_BIG" -ngl "$NLAYER" -sm layer -fa on -c 4096 -npp 128 -ntg 32 -npl 1 \
        > "$RESULTS/m2b.log" 2>&1
    grep -iE "pipeline parallelism enabled" "$RESULTS/m2b.log" \
        && echo "  >> ENABLED (unexpected - the > vs >= gate may have changed)" \
        || echo "  >> correctly OFF at -ngl == n_layer (the silent footgun: always use -ngl 999)"
  fi
  unset CUDA_VISIBLE_DEVICES
else
  echo "  only one visible GPU - skip (run on >=2 GPUs to exercise the gate)"
fi
echo

# =============================================================================
echo "############## M3: CPU hot spots (P2 spawn / P3 mutex / P4 expf) ########"
# =============================================================================
# strace: count clone() syscalls. Per-chunk thread respawn => clones ~ chunks*workers.
if command -v strace >/dev/null 2>&1; then
  echo "--- M3a: strace clone count (expect ~ chunks * workers if P2 not done) ---"
  strace -f -e trace=clone,clone3 -c -o "$RESULTS/m3a.strace" \
      "$PPL" -m "$MODEL" -f "$WIKI" -ngl 999 -fa on -c "$CTX" --chunks "$CHUNKS_FAST" \
      > "$RESULTS/m3a.log" 2>&1
  grep -E "clone|calls" "$RESULTS/m3a.strace" | sed 's/^/    /' || cat "$RESULTS/m3a.strace"
else
  echo "  strace not available - skip M3a"
fi
# perf: top CPU symbols. expf -> P4; __lll_lock/futex -> P3; clone -> P2.
if command -v perf >/dev/null 2>&1; then
  echo "--- M3b: perf record/report top symbols ---"
  perf record -g -o "$RESULTS/m3b.perf.data" -- \
      "$PPL" -m "$MODEL" -f "$WIKI" -ngl 999 -fa on -c "$CTX" --chunks "$CHUNKS_FAST" \
      > "$RESULTS/m3b.log" 2>&1
  if [ -s "$RESULTS/m3b.perf.data" ]; then
    perf report -i "$RESULTS/m3b.perf.data" --stdio 2>/dev/null | head -n 30 | sed 's/^/    /'
    echo "  >> look for: expf/exp (P4 target), __lll_lock_wait/futex (P3 mutex), clone (P2 spawn)"
  else
    echo "  perf produced no data (perf_event_paranoid? try: sysctl kernel.perf_event_paranoid=1)"
  fi
else
  echo "  perf not available - skip M3b"
fi
echo

# =============================================================================
echo "############## M4: GPU timeline (P5 idle gap / B1 warmup artifact) ######"
# =============================================================================
if command -v nsys >/dev/null 2>&1; then
  echo "--- M4a: nsys on perplexity (P5: GPU idle between decode and NLL post-proc?) ---"
  nsys profile -o "$RESULTS/m4a_ppl" --force-overwrite true \
      "$PPL" -m "$MODEL" -f "$WIKI" -ngl 999 -fa on -c "$CTX" --chunks "$CHUNKS_FAST" \
      > "$RESULTS/m4a.log" 2>&1
  nsys stats --report gpukernsum "$RESULTS/m4a_ppl.nsys-rep" 2>/dev/null | head -n 20 | sed 's/^/    /' \
      || echo "    (open $RESULTS/m4a_ppl.nsys-rep in Nsight Systems; gaps on the GPU row = P5 opportunity)"
else
  echo "  nsys not available (ml nvhpc / cuda) - skip M4a/M4c"
fi

# B1 shows up in the TIMING TABLE itself (first cell anomalous vs neighbours) - no nsys needed.
echo "--- M4b: batched-bench wide sweep table (B1: is the first cell out of line?) ---"
"$BB" -m "$MODEL" -ngl 999 -sm layer -fa on -c "$BB_CTX" \
    -npp 512,1024,2048 -ntg 128 -npl 1,2,4,8 > "$RESULTS/m4b.log" 2>&1
grep -E "^\|" "$RESULTS/m4b.log" | sed 's/^/    /' || tail -n 8 "$RESULTS/m4b.log"
echo "  >> compare S_PP/S_TG of the FIRST row vs adjacent rows at the same shape;"
echo "     an anomalously low first row = one-time warmup landing in a timed cell (B1)."

if command -v nsys >/dev/null 2>&1; then
  echo "--- M4c: nsys on a small batched-bench sweep (see the graph-capture in the timeline) ---"
  nsys profile -o "$RESULTS/m4c_bb" --force-overwrite true \
      "$BB" -m "$MODEL" -ngl 999 -sm layer -fa on -c 4096 -npp 512,2048 -ntg 128 -npl 1,4 \
      > "$RESULTS/m4c.log" 2>&1
  echo "    (open $RESULTS/m4c_bb.nsys-rep; extra cuGraph/cublasLt time on the FIRST cell = B1)"
fi
echo

# =============================================================================
echo "############## M5: does the build have a vectorised expf? (P4 feasibility)"
# =============================================================================
if command -v objdump >/dev/null 2>&1; then
  ZGV=$(objdump -d "$PPL" 2>/dev/null | grep -c "_ZGV")
  EXPF=$(objdump -T "$PPL" 2>/dev/null | grep -iE "_ZGV.*expf" | head -3)
  echo "  vectorised math symbols (_ZGV*): $ZGV"
  [ -n "$EXPF" ] && echo "$EXPF" | sed 's/^/    /'
  if [ "$ZGV" -gt 0 ] 2>/dev/null; then
    echo "  >> libmvec present: P4 (#pragma omp simd over expf) CAN vectorise. Worth it if M1 says CPU-bound."
  else
    echo "  >> NO vectorised expf linked: P4 as written would be a no-op. Need -fopenmp + -lmvec/-lm and"
    echo "     -DGGML_NATIVE=ON, OR accept a fast approximate expf (breaks the MUST-MATCH gate)."
  fi
else
  echo "  objdump not available - skip M5"
fi
echo

echo "done -> $RESULTS"
echo "DECISION ORDER:"
echo "  1. M2 first: if pipeline parallelism is off on multi-GPU, fix the run command before anything else."
echo "  2. M1: if CPU post-proc share is small, do NOT spend effort on P2/P4/P5."
echo "  3. If M1 says CPU-bound: M3 tells you which of P2(spawn)/P3(mutex)/P4(expf) dominates,"
echo "     M5 tells you whether P4 can actually vectorise, M4a tells you whether P5 has slack to exploit."
echo "  4. M4b is independent: it justifies B1 (per-shape warmup) for trustworthy bench numbers."
