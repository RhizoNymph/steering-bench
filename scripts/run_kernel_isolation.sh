#!/usr/bin/env bash
# run_kernel_isolation.sh — A/B benchmark isolating the steering Triton
# kernel's contribution under CUDA graph capture.
#
# Runs bench_cuda_graphs.py twice against the same vLLM revision:
#
#   tag=kernel-on    VLLM_STEERING_FORCE_EAGER unset → Triton kernel
#   tag=kernel-off   VLLM_STEERING_FORCE_EAGER=1     → PyTorch eager body
#                                                     inside the opaque op
#
# The op stays opaque to torch.compile in both runs, the warmup loop fires
# both paths, and the rest of the steering perf stack is identical. The
# only thing that changes between runs is the body of the registered
# ``torch.ops.vllm.apply_steering`` op.
#
# Two workload shapes are exercised:
#
#   decode-heavy : prompt_len=64,   max_tokens=128
#   prefill-heavy: prompt_len=2048, max_tokens=8
#
# The kernel's bandwidth win lives at large N, so the prefill workload is
# where the kernel should actually be doing visible work; the decode
# workload should show very little kernel-specific delta because per-call
# launch overhead disappears under graph capture and N=1 traffic is tiny.
#
# Env knobs:
#   MODEL=...    default google/gemma-3-4b-it
#   OUT=...      default results/3090/kernel-isolation
#   PY=...       default .venv/bin/python (resolved against repo root)
#   WARMUP=...   default 10
#   ITERS=...    default 30
#   BATCH=...    default "1,4,8"
#
# Usage:
#   cd /path/to/steering-bench
#   ./scripts/run_kernel_isolation.sh 2>&1 | tee run_kernel_isolation.log
#   ./scripts/run_kernel_isolation.sh --analyze-only    # only re-print summary

set -euo pipefail

MODEL="${MODEL:-google/gemma-3-4b-it}"
OUT="${OUT:-results/3090/kernel-isolation}"
PY="${PY:-.venv/bin/python}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-30}"
BATCH="${BATCH:-1,4,8}"

ANALYZE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --analyze-only) ANALYZE_ONLY=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$OUT/decode" "$OUT/prefill"

run_one() {
  local tag="$1" force_eager="$2" subdir="$3" prompt_len="$4" max_tokens="$5"
  printf '\n>>> [%(%H:%M:%S)T] %s  prompt_len=%s max_tokens=%s\n' \
      -1 "$tag" "$prompt_len" "$max_tokens"
  if [[ "$force_eager" == "1" ]]; then
    VLLM_STEERING_FORCE_EAGER=1 \
      "$PY" scripts/bench_cuda_graphs.py \
        --model "$MODEL" \
        --output-dir "$OUT/$subdir/" --tag "$tag" \
        --warmup "$WARMUP" --iters "$ITERS" \
        --batch-sizes "$BATCH" \
        --prompt-len "$prompt_len" --max-tokens "$max_tokens"
  else
    "$PY" scripts/bench_cuda_graphs.py \
      --model "$MODEL" \
      --output-dir "$OUT/$subdir/" --tag "$tag" \
      --warmup "$WARMUP" --iters "$ITERS" \
      --batch-sizes "$BATCH" \
      --prompt-len "$prompt_len" --max-tokens "$max_tokens"
  fi
}

if [[ "$ANALYZE_ONLY" == "0" ]]; then
  # Sanity: verify the env var actually flips the dispatch flag at import.
  printf '\n=== sanity: VLLM_STEERING_FORCE_EAGER toggle ===\n'
  "$PY" -c "
import os
os.environ.pop('VLLM_STEERING_FORCE_EAGER', None)
from vllm.model_executor.layers import steering
assert steering._FORCE_EAGER_BODY is False, 'expected False without env var'
print('  unset  -> _FORCE_EAGER_BODY=False  OK')
"
  VLLM_STEERING_FORCE_EAGER=1 "$PY" -c "
from vllm.model_executor.layers import steering
assert steering._FORCE_EAGER_BODY is True, 'expected True with env var=1'
print('  set=1  -> _FORCE_EAGER_BODY=True   OK')
"

  # Decode-heavy: small prompt, lots of decode steps. Kernel's launch-
  # overhead win disappears under graph capture, so we expect kernel-on
  # vs kernel-off to be near-flat here.
  run_one kernel-on  0 decode 64   128
  run_one kernel-off 1 decode 64   128

  # Prefill-heavy: long prompt, few decode steps. Each prefill step calls
  # apply_steering on ~2k tokens, so the kernel's bandwidth win (one
  # streaming pass over hidden_states vs the eager body's two passes)
  # should materialize at wall-clock here.
  run_one kernel-on  0 prefill 2048 8
  run_one kernel-off 1 prefill 2048 8
fi

printf '\n\n=========================================================\n'
printf '  Kernel-isolation summary\n'
printf '=========================================================\n'
"$PY" scripts/analyze_kernel_isolation.py --root "$OUT"
