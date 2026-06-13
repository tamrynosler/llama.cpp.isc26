#!/usr/bin/env bash
#SBATCH --job-name=dp-watchdog
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=dp-watchdog-%j.log
#
# S18 (watchdog, GPU verification): confirm the best-effort per-job timeout bounds a REAL decode
# loop on CUDA. The orchestrator arms a per-job deadline; a long generation checks deadline_expired()
# BETWEEN llama_decode steps and stops when it trips. This is the only lever on CUDA, because the
# CUDA backend does NOT poll the ggml abort callback (only CPU/Metal do) - so a wedged single kernel
# still can't be force-killed, but a multi-step generation IS bounded.
#
# The smoke test's "B2: watchdog decode" sub-test does exactly this: arm a 30 ms deadline, decode in
# a loop, and assert it stops early (steps>0, stopped_by_deadline=1, counted as 1 failure). Running
# the smoke binary on an H100 is the cluster proof (locally it runs on Metal; CPU-only build is
# inconclusive for decode and relies on the B spin-loop test for the mechanism).
#
# Cluster notes: NO GRES - select the GPU with CUDA_VISIBLE_DEVICES. Modules via lmod. The smoke
# target is gated on -DLLAMA_BUILD_TESTS=ON, so build it ONCE on a login/interactive node first
# (this batch job only runs the pre-built binary, like the other scripts):
#   cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target llama-orchestrator-smoke -j$(nproc)
#
# Usage:
#   sbatch tools/orchestrator/slurm/dp-watchdog.slurm

set -uo pipefail

REPO="$HOME/llama.cpp.isc26"
MODEL="$HOME/models/qwen2-0_5b-instruct-q4_k_m.gguf"   # any small GGUF works (decode content is irrelevant)
BUILD="$REPO/build"                                    # the CUDA build dir
SMOKE="$BUILD/bin/llama-orchestrator-smoke"

ml gcc cuda

echo "host=$(hostname)"
nvidia-smi -L 2>/dev/null || true
( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
echo

# expects a pre-built smoke binary (build it once on a login node - see header). this batch job
# only RUNS it, so the build environment / cmake module is not needed here.
if [ ! -x "$SMOKE" ]; then
  echo "smoke binary not found at $SMOKE"
  echo "build it once with: cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release"
  echo "                    cmake --build build --target llama-orchestrator-smoke -j\$(nproc)"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0   # one H100 is enough; "0,0" = two replicas on it

echo "=== smoke (2 replicas on 1 H100; watch the 'watchdog decode' line) ==="
LOG="watchdog-smoke-${SLURM_JOB_ID:-local}.log"
"$SMOKE" "$MODEL" 0,0 99 > "$LOG" 2>&1
rc=$?

grep -iE "watchdog decode|timeout drain|PASS|FAIL" "$LOG" || true
echo
if grep -q "stopped_by_deadline=1" "$LOG" && grep -q "\[smoke\] PASS" "$LOG"; then
  echo ">> WATCHDOG (CUDA): PASS - the deadline bounded a real GPU decode loop between steps."
else
  echo ">> WATCHDOG (CUDA): CHECK $LOG (smoke rc=$rc) - expected stopped_by_deadline=1 and [smoke] PASS."
fi
echo
echo "note: the abort callback is NOT polled by CUDA (verified by grep of the backend); the"
echo "      between-step deadline_expired() check above is the real lever. A single wedged CUDA"
echo "      kernel/driver call still cannot be force-killed - drain() stays blocking by design."
