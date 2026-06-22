#!/usr/bin/env bash
#SBATCH --job-name=dp-spec-cluster
#SBATCH --partition=workers
#SBATCH --nodes=3
#SBATCH --nodelist=cn001,cn002,cn003
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=dp-spec-cluster-%j.log
#
# PARTITION: cn001 lives only in the 'workers' partition (the default 'workers_not_build' holds
# cn[002-003] only), so a run touching cn001 MUST use --partition=workers, and cn001 must be free
# (not held by an interactive allocation). Drop to --nodes=2 --nodelist=cn002,cn003 if cn001 is busy.
#
# C++-NATIVE cross-node DATA-PARALLEL SPECULATIVE decoding, STRONG scaling. This lifts the single-node
# dp-speculative.slurm one level up via the MPI cluster_link (same mechanism as dp-batched-bench-cluster).
# srun launches one rank per node; each rank builds its own node-local pool of DP (target+draft) pairs
# (one pair per local GPU), the ranks barrier() to align the start, each runs its STATIC stripe of the
# corpus (chunk i -> node i % n_nodes; NO cross-node work-stealing, intra-node stealing still balances
# the local pairs), and gather() per-node throughput to rank 0, which prints the cluster table. NO python,
# NO inter-node traffic in the inference path (only a start barrier + a kilobyte stats reduce per run).
#
# STRONG scaling = a fixed corpus of M distinct prompts is divided across ALL nodes x GPUs. Compare:
#   (A) 1 GPU,            -dp 1 --dp-chunk-chars C  -> serial baseline: all M prompts on one GPU (T1)
#   (B) N nodes x DP GPU, -dp DP (cluster)          -> M prompts spread across N*DP pairs   (T_cluster)
# REAL speedup = T1 / T_cluster (ideal = N*DP). The spec speedup is inside each stream; DP x nodes
# multiplies the batch throughput.
#
# REQUIRES the MPI build (cluster_link is gated behind -DLLAMA_ORCH_MPI; default OFF). Build once on a
# login/compute node (shared NFS home -> visible on every node):
#   ml gcc hpcx cuda                    # ORDER MATTERS: hpcx requires cuda already loaded
#   cmake -B build-mpi -DGGML_CUDA=ON -DLLAMA_ORCH_MPI=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-mpi -j$(nproc)
#
# Cluster notes: NO GRES - pick GPUs with CUDA_VISIBLE_DEVICES, no "#SBATCH --gres". Modules via lmod.
#
# Usage: build (above), fill in the variables, then:
#   sbatch tools/orchestrator/slurm/dp-speculative-cluster.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-mpi/bin/llama-speculative"     # the -DLLAMA_ORCH_MPI=ON build
MODELS_DIR=/shared/llms/models
DATASET=/shared/llms/datasets/wikitext-2-raw/wiki.test.raw

# target|draft|label  (Qwen2 tokenizer family - vocab must match between target and draft)
SPEC_PAIRS=(
  "Qwen2.5-14B-Instruct-Q4_K_M.gguf|qwen2.5-0.5b-instruct-q4_k_m.gguf|14B+0.5B"
)
DMAX_SWEEP=(4 8 16)   # --spec-draft-n-max values to sweep (tokens drafted per step)
DMIN=2                # --spec-draft-n-min
N_PREDICT=256         # tokens generated per prompt
DP=4                  # replicas PER NODE (one target+draft pair per local GPU); cluster total = DP * #nodes
SEED=42               # fixed seed (reproducible; distinct prompts still diverge - that's the point)

CHUNK_CHARS=4096      # bytes per shard -> one distinct prompt each
# enough distinct prompts to tile the whole cluster evenly: NUM_CHUNKS = #nodes * DP * shards-per-GPU.
# 3 nodes x 4 GPUs x 4 = 48 (16 per node, 4 per local pair). Keep it a multiple of (#nodes * DP).
NUM_CHUNKS=48
CORPUS_CHARS=$(( CHUNK_CHARS * NUM_CHUNKS ))

CTX=8192
BATCH=2048
UBATCH=512
NGL=99                # target GPU layers
NGLD=99               # draft GPU layers (-ngld)
CTK=f16
CTV=f16
MPI_PMI=pmix          # srun PMI plugin (cluster offers: pmi2, pmix). HPC-X speaks pmix.
# -----------------------------------------------------------------------------

ml gcc cuda hpcx      # ORDER MATTERS: hpcx requires cuda already loaded. pin: ml gcc/15.2 cuda/13.3 hpcx/2.26

# --export=ALL pushes the loaded module env (CUDA + HPC-X libs, all on shared NFS) to every srun task,
# so the remote rank can find libcudart / libmpi. --kill-on-bad-exit=1 tears the step down if any rank
# dies instead of leaving the survivors deadlocked in MPI_Init/pmix.
SRUN_OPTS="--export=ALL --kill-on-bad-exit=1"
NNODES="${SLURM_NNODES:-3}"

RESULTS="$REPO/results/dp-spec-cluster-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

# slice a CORPUS from the dataset; the tool slices it into NUM_CHUNKS distinct prompts of CHUNK_CHARS.
CORPUS_FILE="$(mktemp /tmp/dp_spec_cluster_corpus_XXXX.txt)"
head -c "$CORPUS_CHARS" "$DATASET" > "$CORPUS_FILE"
trap 'rm -f "$CORPUS_FILE"' EXIT

echo "launch host=$(hostname)  nodelist=${SLURM_JOB_NODELIST:-?}  nnodes=${NNODES}  dp/node=$DP  cluster pairs=$(( NNODES * DP ))"
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "corpus: first ${CORPUS_CHARS} bytes of $(basename "$DATASET") -> ${NUM_CHUNKS} x ${CHUNK_CHARS}-byte prompts"
echo "srun --mpi=$MPI_PMI  (fallback if it errors: set MPI_PMI=pmi2, or use mpirun)"
echo

# pre-flight: confirm the srun<->HPC-X pmix handshake before the (long) sweep. Cheap (no model);
# only runs if the test binary was built (-DLLAMA_BUILD_TESTS=ON). Fails the job fast on a bad PMI.
PREFLIGHT="$REPO/build-mpi/bin/llama-orchestrator-cluster-test"
if [[ -x "$PREFLIGHT" ]]; then
  echo "=== pmix pre-flight: cluster_link on ${NNODES} nodes (--mpi=$MPI_PMI) ==="
  if ! srun --mpi="$MPI_PMI" $SRUN_OPTS --nodes="$NNODES" --ntasks-per-node=1 "$PREFLIGHT"; then
    echo "FATAL: cluster_link pre-flight failed under --mpi=$MPI_PMI; try MPI_PMI=pmi2." >&2
    exit 1
  fi
  echo
fi

# run <label> <logfile> <command...> - capture full output, echo the strong-scaling result lines.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then
    grep -E "strong scaling|prompts ok|gen tokens|aggregate decode|real .*speedup|^\| *[0-9A-Z]" "$log" \
      || echo "  (no result lines - see $log)"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

# baseline wall comes from the 1-GPU strong run ("... in 12.185s wall ..."); cluster wall comes from
# rank 0's CL summary line ("... slowest-node wall 3.114s ...").
wall_of()         { sed -nE 's/.* in ([0-9.]+)s wall.*/\1/p' "$1" | head -1; }
cluster_wall_of() { sed -nE 's/.*slowest-node wall ([0-9.]+)s.*/\1/p' "$1" | head -1; }

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
    cl_log="$RESULTS/${plabel}.dmax${dmax}.cluster.log"

    # (A) 1 GPU serial baseline: all M prompts on one GPU. Run the binary directly (no srun) -> MPI
    # world size 1 -> single-node path; CUDA_VISIBLE_DEVICES pins it to one card.
    export CUDA_VISIBLE_DEVICES=0
    run "SPEC strong | $plabel | dmax=$dmax | 1 GPU serial baseline" \
        "$base_log" \
        "$BIN" "${common[@]}" --data-parallel 1

    # (B) cluster: M prompts striped across nodes, work-stolen within each node across DP pairs.
    # srun spawns one rank per node; the binary auto-detects cluster mode (MPI world > 1) and rank 0
    # prints the per-NODE + CL table.
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    run "SPEC strong | $plabel | dmax=$dmax | ${NNODES} nodes x ${DP} GPUs cluster" \
        "$cl_log" \
        srun --mpi="$MPI_PMI" $SRUN_OPTS --nodes="$NNODES" --ntasks-per-node=1 \
            "$BIN" "${common[@]}" --data-parallel "$DP" --dp-devices 0,1,2,3

    # real cluster speedup = serial wall / cluster slowest-node wall
    t1="$(wall_of "$base_log")"; tc="$(cluster_wall_of "$cl_log")"
    awk -v a="$t1" -v b="$tc" -v ideal="$(( NNODES * DP ))" 'BEGIN{
      if (a != "" && b != "" && b+0 > 0)
        printf "  >> REAL CLUSTER SPEEDUP (1 GPU / %d pairs): %ss / %ss = %.2fx (ideal %d)\n\n", ideal, a, b, a/b, ideal;
      else print "  >> speedup: could not parse walls (see logs)\n";
    }'
  done
done

echo "done -> $RESULTS"
echo "headline: per (pair,dmax) the 'REAL CLUSTER SPEEDUP' = 1-GPU-serial wall / cluster slowest-node wall"
echo "          over the SAME ${NUM_CHUNKS} distinct prompts (ideal = ${NNODES} nodes x ${DP} = $(( NNODES * DP )))."
echo "          In each *.cluster.log the per-NODE rows are each one node's local DP result; the CL row"
echo "          sums tokens across nodes over the slowest-node wall. accept% should match the 1-GPU run."
