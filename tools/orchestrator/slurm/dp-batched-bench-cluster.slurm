#!/usr/bin/env bash
#SBATCH --job-name=bb-dp-cluster
#SBATCH --partition=workers
#SBATCH --nodes=2
#SBATCH --nodelist=cn001,cn002
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=bb-dp-cluster-%j.log
#
# PARTITION: cn001 lives only in the 'workers' partition (the default 'workers_not_build'
# holds cn[002-003] only), so a 2-node run touching cn001 MUST use --partition=workers.
# cn001 must also be free (not held by an interactive allocation) for this to schedule.
#
# C++-NATIVE cross-node data-parallel batched-bench (S19 Phase 2). This replaces the srun+python
# prototype (dp-multinode-srun.slurm + aggregate-multinode.py, job 453): coordination + aggregation
# now happen IN-BINARY via the MPI cluster_link. srun launches one rank per node; each rank builds
# its own single-node DP pool over its 4 local GPUs, the ranks barrier() to align the start, run the
# byte-identical node-local sweep, and gather() per-config throughput to rank 0, which prints the
# cluster table. NO python, NO inter-node traffic in the inference path (only a start barrier + a
# kilobyte stats reduce per run) - that node-local property is what preserves the measured scaling.
#
# cn003 is down: this uses cn001,cn002 (2 nodes) -> ideal node scaling = 2.00x (matches job 453).
#
# OUTPUT IS SELF-VERIFYING. For each config the binary prints:
#   - one NODE row per node  == that node's single-node DP "ALL" row  -> compare to job 446
#   - one CL  (cluster) row  == tokens summed across nodes / slowest node span -> compare to job 453
#
# REQUIRES the MPI build (cluster_link is gated behind -DLLAMA_ORCH_MPI; default OFF). Build once on
# a login/compute node (shared NFS home -> visible on every node):
#   ml gcc hpcx cuda                    # HPC-X provides MPI (Open MPI + UCX); pmix-capable
#   cmake -B build-mpi -DGGML_CUDA=ON -DLLAMA_ORCH_MPI=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-mpi -j$(nproc)
#
# Cluster notes: NO GRES - pick GPUs with CUDA_VISIBLE_DEVICES, no "#SBATCH --gres". Modules via
# lmod. Home is shared (NFS), so one build + one results dir are visible on every node.
#
# Usage: build (above), fill in the variables, then:
#   sbatch tools/orchestrator/slurm/dp-batched-bench-cluster.slurm

set -uo pipefail

# ---- fill these in ----------------------------------------------------------
REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-mpi/bin/llama-batched-bench"   # the -DLLAMA_ORCH_MPI=ON build
MODELS=(
  "$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"        # SAME 5 models as job 446 / 453
  "$HOME/models/qwen2.5-1.5b-instruct-q5_k_m.gguf"
  "$HOME/models/qwen2.5-3b-instruct-q5_k_m.gguf"
  "$HOME/models/OLMo-2-1124-13B-Instruct-Q4_0.gguf"
  "$HOME/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)
CTX=65536            # per-replica context; must hold the largest config: max(PL)*(PP+TG) <= CTX
NGL=99               # offload all layers to the GPU
DP=4                 # replicas PER NODE (one per local GPU); cluster total = DP * #nodes
NBATCH=4096          # logical batch (-b)
UBATCH=2048          # physical micro-batch (-ub)
PP=512               # prompt lengths to sweep (comma-separated ok)
TG=128               # generation lengths
PL=1,2,4,8,16,32,64  # parallel sequences per replica (the main throughput lever)
MPI_PMI=pmix         # srun PMI plugin (cluster offers: pmi2, pmix). HPC-X speaks pmix.
# -----------------------------------------------------------------------------

ml gcc hpcx cuda     # pin if needed: ml gcc/15.2 hpcx/2.26 cuda/13.3

RESULTS="$REPO/results/dp-cluster-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

echo "launch host=$(hostname)  nodelist=${SLURM_JOB_NODELIST:-?}  nnodes=${SLURM_NNODES:-?}  dp/node=$DP"
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "srun --mpi=$MPI_PMI  (fallback if it errors: set MPI_PMI=pmi2, or use mpirun)"
echo

# each node selects its 4 local GPUs (CUDA_VISIBLE_DEVICES is per-node-local; srun propagates env).
export CUDA_VISIBLE_DEVICES=0,1,2,3

# pre-flight: confirm the srun<->HPC-X pmix handshake before the (long) sweep. Cheap (no model);
# only runs if the test binary was built (-DLLAMA_BUILD_TESTS=ON). Fails the job fast on a bad PMI.
PREFLIGHT="$REPO/build-mpi/bin/llama-orchestrator-cluster-test"
if [[ -x "$PREFLIGHT" ]]; then
  echo "=== pmix pre-flight: cluster_link on ${SLURM_NNODES:-2} nodes (--mpi=$MPI_PMI) ==="
  if ! srun --mpi="$MPI_PMI" --nodes="${SLURM_NNODES:-2}" --ntasks-per-node=1 "$PREFLIGHT"; then
    echo "FATAL: cluster_link pre-flight failed under --mpi=$MPI_PMI; try MPI_PMI=pmi2." >&2
    exit 1
  fi
  echo
fi

# run <label> <logfile> <command...> - capture full output, echo the cluster table rows.
run() {
  local label="$1"; shift
  local log="$1";   shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then
    grep -E "^\|" "$log" || echo "  (no table rows - see $log)"
  else
    echo "FAILED ($label) - see $log"; tail -n 20 "$log"
  fi
  echo
}

# one srun per model = one fresh MPI world per model (clean MPI_Init/Finalize each time). srun spawns
# one rank per allocated node; the binary auto-detects cluster mode (MPI world size > 1) and rank 0
# prints the aggregated table. --data-parallel drives the per-node replica pool exactly as job 446.
for M in "${MODELS[@]}"; do
  name="$(basename "$M" .gguf)"
  run "CLUSTER | $name | ${SLURM_NNODES:-?} nodes x $DP GPUs | --data-parallel $DP" \
      "$RESULTS/${name}.cluster.log" \
      srun --mpi="$MPI_PMI" --nodes="${SLURM_NNODES:-2}" --ntasks-per-node=1 \
          "$BIN" -m "$M" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
                 -npp "$PP" -ntg "$TG" -npl "$PL" \
                 --data-parallel "$DP" --dp-devices 0,1,2,3
done

echo "done -> $RESULTS"
echo "verify: in each *.cluster.log, the per-NODE rows should match that model's single-node DP 'ALL'"
echo "        row from job 446; the 'CL' row's S_e2e t/s should be ~${SLURM_NNODES:-2}x a single node"
echo "        (job 453 measured 2.00x on 2 nodes, python-aggregated - this reproduces it in-binary)."
echo "tip: add --output-format jsonl for machine-readable per-node + cluster (cl_*) fields."
