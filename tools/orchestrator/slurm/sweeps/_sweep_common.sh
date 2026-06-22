#!/usr/bin/env bash
# Shared driver for the six batched-bench tuning sweeps (dense/MoE x prefill/decode/balanced).
#
# A sweep script sources this file, sets a handful of globals, and calls run_sweep. Everything
# below is generic: the model registry, the per-config runner (jsonl out), and the two-phase
# sweep structure (Phase A = one knob at a time at a reference layout; Phase B = orchestration /
# split layout scaling at the baseline knobs). See README.md for the design and analysis recipe.
#
# Globals a sweep script MUST set before calling run_sweep:
#   SWEEP_NAME       label for the results dir, e.g. "dense-prefill"
#   MODELS=(...)     short names from the registry below (one "small" + one "large" per sweep)
#   PP_small TG_small PL_small CTX_small      shapes for models that fit one 80 GB card
#   PP_large TG_large PL_large CTX_large      shapes for models that need a 2-GPU split
#   KNOBS=("label|<args>" ...)                Phase A grid: each entry varies ONE knob off base
#   BASELINE_KNOBS="<args>"                   knobs used during the Phase B layout sweep
#
# Conventions: NO "#SBATCH --gres" on this cluster (no GRES) - GPUs are selected with
# CUDA_VISIBLE_DEVICES. Modules via lmod. batched-bench uses random tokens (no dataset needed).

REPO="${REPO:-$HOME/llama.cpp.isc26}"
BIN="${BIN:-$REPO/build/bin/llama-batched-bench}"

# ---- model registry (Q8 across the board, all under ~/models) --------------------------------
# resolve_model echoes the absolute path to the (first shard of the) gguf, or nothing if absent.
# Sharded models live in a subfolder; we glob the *-00001-of-*.gguf shard (llama loads the rest).
resolve_model() {
  case "$1" in
    qwen14b)       echo "$HOME/models/Qwen2.5-14B-Instruct-Q8_0.gguf" ;;
    mistral-large) ls "$HOME/models/Mistral-Large-Instruct-2407-Q8_0"/*-00001-of-*.gguf 2>/dev/null | head -1 ;;
    deepseek-moe)  ls "$HOME/models/deepseek-moe-16b-base.Q8_0.gguf" 2>/dev/null \
                     || ls "$HOME/models"/deepseek-moe-16b-base*Q8_0*-00001-of-*.gguf 2>/dev/null | head -1 ;;
    scout)         ls "$HOME/models/Llama-4-Scout-17B-16E-Q8/Q8_0"/*-00001-of-*.gguf 2>/dev/null | head -1 ;;
    *)             echo "" ;;
  esac
}

# small = fits one 80 GB card (DP + oversubscription story); large = needs a 2-GPU split.
model_class() {
  case "$1" in
    qwen14b|deepseek-moe) echo small ;;
    mistral-large|scout)  echo large ;;
    *)                    echo small ;;
  esac
}

# reference layout for the Phase A knob sweep: the smallest layout that can hold the model.
ref_layout() {
  case "$1" in
    small) echo "1gpu;0;" ;;
    large) echo "split1-row;0,1;--dp-devices 0+1 --dp-split row" ;;
  esac
}

# Phase B layouts (label;CUDA_VISIBLE_DEVICES;dp-args), one per line.
#   small: stock 1-GPU -> DP 2/4 -> oversubscription (R replicas per GPU on all 4).
#   large: 2-GPU split (1 then 2 replicas) x {row,layer}, plus a single 4-GPU split.
layouts_for_class() {
  case "$1" in
    small) cat <<'EOF'
1gpu;0;
dp2;0,1;--data-parallel 2 --dp-devices 0,1
dp4;0,1,2,3;--data-parallel 4 --dp-devices 0,1,2,3
oversub2;0,1,2,3;--dp-num-devices 4 --dp-replicas-per-device 2
oversub4;0,1,2,3;--dp-num-devices 4 --dp-replicas-per-device 4
EOF
    ;;
    large) cat <<'EOF'
split1-row;0,1;--dp-devices 0+1 --dp-split row
split1-layer;0,1;--dp-devices 0+1 --dp-split layer
split2-row;0,1,2,3;--dp-devices 0+1,2+3 --dp-split row
split2-layer;0,1,2,3;--dp-devices 0+1,2+3 --dp-split layer
split4-row;0,1,2,3;--dp-devices 0+1+2+3 --dp-split row
EOF
    ;;
  esac
}

sweep_init() {
  ml gcc cuda            # pin if needed: ml gcc/15.2 cuda/13.1
  RESULTS="results/sweep-${SWEEP_NAME}-${SLURM_JOB_ID:-local}"
  mkdir -p "$RESULTS"
  echo "host=$(hostname)   sweep=$SWEEP_NAME"
  nvidia-smi -L 2>/dev/null || true
  ( cd "$REPO" && echo "fork commit: $(git rev-parse --short HEAD 2>/dev/null)" ) || true
  echo "baseline knobs: $BASELINE_KNOBS"
  echo "results dir:    $RESULTS"
  echo
}

# bench <tag> <batched-bench args...> : run one config, jsonl -> $RESULTS/<tag>.jsonl.
# CUDA_VISIBLE_DEVICES must be exported by the caller. tag must be filesystem-safe.
bench() {
  local tag="$1"; shift
  echo "=== $tag ==="
  echo "    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(basename "$BIN") $*"
  if "$BIN" "$@" --output-format jsonl > "$RESULTS/${tag}.jsonl" 2> "$RESULTS/${tag}.err"; then
    tail -n 5 "$RESULTS/${tag}.jsonl" | sed 's/^/    /'
  else
    echo "  FAILED ($tag) - tail of stderr:"; tail -n 12 "$RESULTS/${tag}.err" | sed 's/^/    /'
  fi
  echo
}

run_sweep() {
  sweep_init
  for name in "${MODELS[@]}"; do
    local path class
    path="$(resolve_model "$name")"
    class="$(model_class "$name")"
    if [ -z "$path" ] || [ ! -f "$path" ]; then
      echo "!! model '$name' not found under ~/models - skipping (did its download finish?)"; echo
      continue
    fi

    # shapes by class (indirect expansion)
    local v PP TG PL CTX
    v="PP_$class";  PP="${!v}"
    v="TG_$class";  TG="${!v}"
    v="PL_$class";  PL="${!v}"
    v="CTX_$class"; CTX="${!v}"
    local shape=(-c "$CTX" -ngl 99 -npp "$PP" -ntg "$TG" -npl "$PL")

    echo "##################################################################"
    echo "# $name  [$class]  ::  $SWEEP_NAME"
    echo "#   $path"
    echo "#   shapes: npp=$PP  ntg=$TG  npl=$PL  ctx=$CTX"
    echo "##################################################################"
    echo

    # ---- Phase A: one knob at a time, at the reference layout --------------------------------
    local rlabel rcvd rargs
    IFS=';' read -r rlabel rcvd rargs <<< "$(ref_layout "$class")"
    for ks in "${KNOBS[@]}"; do
      local klabel kargs
      IFS='|' read -r klabel kargs <<< "$ks"
      export CUDA_VISIBLE_DEVICES="$rcvd"
      bench "${name}.A_knob.${klabel}.at_${rlabel}" -m "$path" "${shape[@]}" $kargs $rargs
    done

    # ---- Phase B: layout scaling, at the baseline knobs --------------------------------------
    while IFS=';' read -r llabel lcvd largs; do
      [ -z "$llabel" ] && continue
      export CUDA_VISIBLE_DEVICES="$lcvd"
      bench "${name}.B_layout.${llabel}" -m "$path" "${shape[@]}" $BASELINE_KNOBS $largs
    done < <(layouts_for_class "$class")
  done

  echo "done -> $RESULTS"
  echo "analyse: jq over $RESULTS/*.jsonl (see tools/orchestrator/slurm/sweeps/README.md)"
}
