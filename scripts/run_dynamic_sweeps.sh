#!/usr/bin/env bash
# Dynamic-steering overhead sweeps for google/gemma-3-4b-it (34 layers,
# hidden 2560). Three presets:
#
#   ladder  full arm ladder x bs {1,8,16,32}, single site (post_block@17).
#           Re-runs the historical ladder + the two new per-request arms.
#   rowmon  off,steer_override,steer_rowmon x bs {1,8,16,32}. steer_override
#           is the no-monitor control; (rowmon - override) IS the per-row
#           in-graph monitor cost.
#   sites   off,steer_sync,steer_override x bs 16 across steered-site counts
#           (spread:1/4/8/17/34 @ post_block) plus one all-hooks config
#           (spread:8 x pre_attn,post_attn,post_block). Monitor site fixed.
#
# Usage:
#   scripts/run_dynamic_sweeps.sh ladder            # one preset
#   scripts/run_dynamic_sweeps.sh rowmon
#   scripts/run_dynamic_sweeps.sh sites
#   scripts/run_dynamic_sweeps.sh all               # all three
#
# Env overrides: MODEL, NUM_LAYERS, OUTLEN, PROMPTLEN, WARMUP, ITERS,
#   GPU_MEM_UTIL, EAGER=1 (disable cudagraphs), TAG, PY (python interpreter),
#   VLLM_COMMIT (recorded in results; auto-detected if unset).
#
# See docs/dynamic_steering_sweep_runbook.md for the node0 deploy recipe.
set -euo pipefail

PRESET="${1:-all}"

MODEL="${MODEL:-google/gemma-3-4b-it}"
NUM_LAYERS="${NUM_LAYERS:-34}"
OUTLEN="${OUTLEN:-64}"
PROMPTLEN="${PROMPTLEN:-64}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
TAG="${TAG:-}"
PY="${PY:-python}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="$HERE/scripts/bench_dynamic_steering.py"

# In-process engine + no flashinfer sampler + fresh compile cache (a stale
# cache bakes the wrong op arity — see the runbook).
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

# node0 ~/.bashrc force-sets HF_HOME to an inaccessible NAS mount on every
# non-interactive bash; pin HF at the real local cache and run offline
# (all weights are cached locally). Override with SWEEP_HF_HOME if needed.
export HF_HOME="${SWEEP_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

EAGER_FLAG=()
[[ "${EAGER:-0}" == "1" ]] && EAGER_FLAG=(--enforce-eager)

COMMIT_FLAG=()
[[ -n "${VLLM_COMMIT:-}" ]] && COMMIT_FLAG=(--vllm-commit "$VLLM_COMMIT")

common=(
  "$PY" "$DRIVER"
  --model "$MODEL" --num-model-layers "$NUM_LAYERS"
  --output-len "$OUTLEN" --prompt-len "$PROMPTLEN"
  --warmup "$WARMUP" --iters "$ITERS" --gpu-mem-util "$GPU_MEM_UTIL"
  "${EAGER_FLAG[@]}" "${COMMIT_FLAG[@]}"
)

run_ladder() {
  echo "=== SWEEP: ladder ==="
  "${common[@]}" \
    --arms off,capture_async,capture_sync,steer_async,steer_sync,steer_dynamic,steer_override,steer_rowmon \
    --batch-sizes 1,8,16,32 \
    --steer-layers 17 --steer-hooks post_block \
    --output-dir results/dynamic_steering/ladder/ \
    --tag "ladder${TAG:+_$TAG}"
}

run_rowmon() {
  echo "=== SWEEP: rowmon ==="
  "${common[@]}" \
    --arms off,steer_override,steer_rowmon \
    --batch-sizes 1,8,16,32 \
    --steer-layers 17 --steer-hooks post_block \
    --output-dir results/dynamic_steering/rowmon/ \
    --tag "rowmon${TAG:+_$TAG}"
}

run_sites() {
  echo "=== SWEEP: sites ==="
  # (steer_layers_spec, steer_hooks, label)
  local configs=(
    "spread:1|post_block|s1"
    "spread:4|post_block|s4"
    "spread:8|post_block|s8"
    "spread:17|post_block|s17"
    "spread:34|post_block|s34"
    "spread:8|pre_attn,post_attn,post_block|s8x3hooks"
  )
  for cfg in "${configs[@]}"; do
    IFS='|' read -r layers hooks label <<<"$cfg"
    echo "--- sites config: layers=$layers hooks=$hooks ---"
    "${common[@]}" \
      --arms off,steer_sync,steer_override \
      --batch-sizes 16 \
      --steer-layers "$layers" --steer-hooks "$hooks" \
      --monitor-layer 17 --monitor-hook post_block \
      --output-dir results/dynamic_steering/sites/ \
      --tag "sites_${label}${TAG:+_$TAG}"
  done
}

case "$PRESET" in
  ladder) run_ladder ;;
  rowmon) run_rowmon ;;
  sites)  run_sites ;;
  all)    run_ladder; run_rowmon; run_sites ;;
  *) echo "unknown preset: $PRESET (want: ladder|rowmon|sites|all)" >&2; exit 2 ;;
esac
