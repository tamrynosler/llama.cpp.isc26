#!/usr/bin/env bash
#SBATCH --job-name=cluster-maxout-mpirun
#SBATCH --partition=workers
#SBATCH --nodes=3
#SBATCH --nodelist=cn001,cn002,cn003
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:18:00
#SBATCH --output=cluster-maxout-mpirun-%j.log
#
# CODE RED: 3 nodes up briefly. ONE run that maxes orchestration over ALL 12 H100s via the MPI
# cluster_link. SAME run as cluster-maxout.slurm, but launched with HPC-X mpirun instead of srun
# (1 rank/node; each rank builds its 4-GPU node-local DP pool; barrier + gather to rank 0, which
# prints per-NODE rows + a CL cluster row). All 4 Q8 models, large batch.
#
#   small models -> 4 replicas/node  = 12 replicas across the cluster (full DP, all 12 GPUs)
#   large models -> 2x(2-GPU split)/node = 6 split replicas across the cluster (all 12 GPUs)
#
# Why a separate script: the cluster_link is a plain MPI program (rank/size come from
# MPI_Comm_rank/size, NOT from SLURM env), so mpirun works too. But mpirun does NOT inherit the
# sbatch task layout or env the way `srun` does, so we MUST reproduce by hand the two things the
# device model assumes:
#   1. EXACTLY one rank per node (--map-by ppr:1:node). The pinning model gives each rank all 4
#      local GPUs and splits them via --dp-devices; >1 rank/node => ranks collide on GPU 0-3.
#   2. CUDA_VISIBLE_DEVICES propagated to every rank (-x CUDA_VISIBLE_DEVICES). srun's --export=ALL
#      did this for free; mpirun forwards nothing unless told.
#
# ORDER: small models FIRST (load in seconds) so a kill still leaves the headline 12-GPU number.
# Each mpirun = one fresh MPI world (one model). Missing models are skipped, never block the window.
# Needs the MPI build:  cmake -B build-mpi -DGGML_CUDA=ON -DLLAMA_ORCH_MPI=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-mpi -j$(nproc)
#   sbatch tools/orchestrator/slurm/sweeps/cluster-maxout-mpirun.slurm

set -uo pipefail

REPO="$HOME/llama.cpp.isc26"
BIN="$REPO/build-mpi/bin/llama-batched-bench"   # the -DLLAMA_ORCH_MPI=ON build
NGL=99
NBATCH=8192          # large logical batch (-b)
UBATCH=4096          # large physical micro-batch (-ub)
PP=512               # prompt length
TG=128               # generation length

# label : class : path-or-glob (first shard for sharded models)
ENTRIES=(
  "Qwen2.5-14B-Q8:small:$HOME/models/Qwen2.5-14B-Instruct-Q8_0.gguf"
  "deepseek-moe-16b-Q8:small:$HOME/models/deepseek-moe-16b-base.Q8_0.gguf"
  "Mistral-Large-2407-Q8:large:$HOME/models/Mistral-Large-Instruct-2407-Q8_0/*-00001-of-*.gguf"
  "Llama-4-Scout-Q8:large:$HOME/models/Llama-4-Scout-17B-16E-Q8/Q8_0/*-00001-of-*.gguf"
)

ml gcc cuda hpcx     # ORDER MATTERS: hpcx needs cuda first. hpcx provides mpirun. pin: ml gcc/15.2 cuda/13.1 hpcx
NN="${SLURM_NNODES:-3}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RESULTS="$REPO/results/cluster-maxout-mpirun-${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS"

# Build an explicit 1-slot-per-node hostfile from the SLURM allocation so mpirun places exactly
# one rank per node and does not auto-fill from the 64 cores it sees per host.
HOSTFILE="$RESULTS/hostfile"
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  scontrol show hostnames "$SLURM_JOB_NODELIST" | sed 's/$/ slots=1/' > "$HOSTFILE"
else
  hostname | sed 's/$/ slots=1/' > "$HOSTFILE"   # local degrade: size()==1
fi

# 1 rank/node + forward the env mpirun does not propagate by default.
MPIRUN_OPTS="--hostfile $HOSTFILE --map-by ppr:1:node --bind-to none \
  -x CUDA_VISIBLE_DEVICES -x PATH -x LD_LIBRARY_PATH"

echo "launch host=$(hostname)  nodelist=${SLURM_JOB_NODELIST:-?}  nnodes=$NN"
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo "mpirun -np $NN --map-by ppr:1:node (hostfile=$HOSTFILE) ; large batch -b $NBATCH -ub $UBATCH ; pp=$PP tg=$TG"
echo "hostfile:"; cat "$HOSTFILE"
echo

# pmix-free pre-flight (cheap, no model) - fail fast on a bad MPI bring-up instead of deadlocking.
PRE="$REPO/build-mpi/bin/llama-orchestrator-cluster-test"
if [[ -x "$PRE" ]]; then
  echo "=== mpirun pre-flight on $NN nodes ==="
  mpirun -np "$NN" $MPIRUN_OPTS "$PRE" \
    || { echo "FATAL: pre-flight failed under mpirun; check 'ml hpcx' and the hostfile." >&2; exit 1; }
  echo
fi

run() {  # label, logfile, command...
  local label="$1"; shift; local log="$1"; shift
  echo "=== $label ==="
  if "$@" > "$log" 2>&1; then grep -E "^\|" "$log" || echo "  (no rows - see $log)"
  else echo "FAILED ($label) - tail:"; tail -n 15 "$log"; fi
  echo
}

for E in "${ENTRIES[@]}"; do
  IFS=':' read -r LBL CLS GLOB <<< "$E"
  M="$(ls $GLOB 2>/dev/null | head -1)"
  if [[ -z "$M" || ! -f "$M" ]]; then echo "skip $LBL (not found: $GLOB)"; echo; continue; fi

  if [[ "$CLS" == small ]]; then
    CTX=49152; PL="1,16,64"; DP_ARGS=(--data-parallel 4 --dp-devices 0,1,2,3)
    DESC="$NN nodes x 4 replicas = $((NN*4)) GPUs (full DP)"
  else
    CTX=8192;  PL="1,4";     DP_ARGS=(--dp-devices 0+1,2+3 --dp-split row)
    DESC="$NN nodes x 2 split-replicas = $((NN*4)) GPUs (2-GPU split)"
  fi

  run "CLUSTER | $LBL | $DESC" "$RESULTS/${LBL}.cluster.log" \
      mpirun -np "$NN" $MPIRUN_OPTS \
          "$BIN" -m "$M" -c "$CTX" -b "$NBATCH" -ub "$UBATCH" -ngl "$NGL" \
                 -npp "$PP" -ntg "$TG" -npl "$PL" "${DP_ARGS[@]}"
done

echo "done -> $RESULTS"
echo "headline: each *.cluster.log 'CL' row S_e2e t/s should be ~${NN}x one node (3-node proof)."
