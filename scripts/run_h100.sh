#!/usr/bin/env bash
# run_h100.sh — full H100 validation session for the steering-bench suite.
#
# Runs everything needed to produce a canonical H100 dataset for the RFC
# comment and PR. Writes all output under results/h100/ so existing 3090
# results in results/* are left untouched.
#
# The script is idempotent at the file level (each benchmark writes a
# timestamped JSON) and resumable at the step level via env vars:
#
#   DO_ENV=0            skip env check
#   DO_CORRECTNESS=0    skip correctness verification
#   DO_MICRO=0          skip microbenchmarks
#   DO_4B=0             skip all Gemma-3-4B runs
#   DO_4B_SYSTEM=0      skip 4B system benchmarks only
#   DO_4B_ABLATION=0    skip 4B ablations only
#   DO_4B_SERVING=0     skip 4B serving benchmark only
#   DO_27B=0            skip all Gemma-3-27B runs
#   DO_27B_SYSTEM=0     skip 27B system benchmarks only
#   DO_27B_ABLATION=0   skip 27B ablations only
#   DO_27B_SERVING=0    skip 27B serving benchmark only
#   DO_ANALYZE=0        skip the final aggregation step
#
#   MODEL_SMALL=...     default google/gemma-3-4b-it
#   MODEL_LARGE=...     default google/gemma-3-27b-it
#   OUT=...             default results/h100
#   TAG=...             default canonical-h100
#   PY=...              default .venv/bin/python
#   SHAREGPT_PATH=...   if set, serving benchmark uses ShareGPT instead of synthetic
#   RFC_WARMUP=...      warmup iters for RFC-critical benchmarks (default 10)
#   RFC_ITERS=...       measured iters for RFC-critical benchmarks (default 30)
#
# Usage:
#   cd /path/to/steering-bench
#   ./scripts/run_h100.sh 2>&1 | tee run_h100.log
#
#   # preview without running anything
#   ./scripts/run_h100.sh --dry-run
#   DRY_RUN=1 ./scripts/run_h100.sh
#
# Expected wall clock on a single H100 80GB: ~5–6 hours total.

set -euo pipefail

# ─── Arg parsing (minimal — just --dry-run / -n) ─────────────────────────
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      sed -n '2,36p' "$0"
      exit 0
      ;;
    *)
      printf 'unknown arg: %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done
: "${DRY_RUN:=0}"

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_SMALL="${MODEL_SMALL:-google/gemma-3-4b-it}"
MODEL_LARGE="${MODEL_LARGE:-google/gemma-3-27b-it}"
OUT="${OUT:-results/h100}"
TAG="${TAG:-canonical-h100}"
PY="${PY:-.venv/bin/python}"
SHAREGPT_PATH="${SHAREGPT_PATH:-}"

# Bumped warmup/iter counts for RFC-critical benchmarks. Rental H100 instances
# typically do not allow clock-locking (no sudo / no privileged nvidia-smi),
# so the GPU is free to boost/throttle. More iterations averages out the
# variance without needing clock pinning.
RFC_WARMUP="${RFC_WARMUP:-10}"
RFC_ITERS="${RFC_ITERS:-30}"

: "${DO_ENV:=1}"
: "${DO_CORRECTNESS:=1}"
: "${DO_MICRO:=1}"
: "${DO_4B:=1}"
: "${DO_4B_SYSTEM:=1}"
: "${DO_4B_ABLATION:=1}"
: "${DO_4B_SERVING:=1}"
: "${DO_27B:=1}"
: "${DO_27B_SYSTEM:=1}"
: "${DO_27B_ABLATION:=1}"
: "${DO_27B_SERVING:=1}"
: "${DO_ANALYZE:=1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p \
    "$OUT/micro" \
    "$OUT/vllm" \
    "$OUT/ablation" \
    "$OUT/serving" \
    "$OUT/summary"
fi

STEP_COUNT=0
STEPS_PLANNED=()

# ─── Logging helpers ─────────────────────────────────────────────────────
section() {
  printf '\n\n=========================================================\n'
  printf '  %s\n' "$1"
  printf '=========================================================\n'
}

run_step() {
  local name="$1"
  shift
  STEP_COUNT=$((STEP_COUNT + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    STEPS_PLANNED+=("$name")
    printf '\n[%3d] %s\n' "$STEP_COUNT" "$name"
    printf '      %s\n' "$*"
    return 0
  fi
  local t0
  t0="$(date +%s)"
  printf '\n>>> [%(%H:%M:%S)T] START %s\n' -1 "$name"
  printf '    cmd: %s\n' "$*"
  if "$@"; then
    local t1
    t1="$(date +%s)"
    printf '>>> [%(%H:%M:%S)T] DONE  %s (%ds)\n' -1 "$name" "$((t1 - t0))"
  else
    local rc=$?
    printf '>>> [%(%H:%M:%S)T] FAIL  %s (rc=%d)\n' -1 "$name" "$rc"
    return "$rc"
  fi
}

SHARED_SERVING_ARGS=()
if [[ -n "$SHAREGPT_PATH" ]]; then
  SHARED_SERVING_ARGS+=(--sharegpt-path "$SHAREGPT_PATH")
fi

# ─── 1. Environment check ────────────────────────────────────────────────
if [[ "$DO_ENV" == "1" ]]; then
  section "environment"
  run_step "gpu+python info" "$PY" - <<'PY'
import sys, torch, platform
print(f"python      : {sys.version.split()[0]}")
print(f"platform    : {platform.platform()}")
print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"capability  : {torch.cuda.get_device_capability(0)}")
    print(f"mem total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
try:
    import vllm
    print(f"vllm        : {vllm.__version__}")
except Exception as e:
    print(f"vllm        : IMPORT FAILED: {e}")
    sys.exit(2)
PY
fi

# ─── 2. Correctness verification (fail fast, cheap) ──────────────────────
if [[ "$DO_CORRECTNESS" == "1" ]]; then
  section "correctness"
  run_step "verify_correctness (4B)" \
    "$PY" scripts/verify_correctness.py --model "$MODEL_SMALL"
fi

# ─── 3. Microbenchmarks (fast, no model serving) ─────────────────────────
if [[ "$DO_MICRO" == "1" ]]; then
  section "microbenchmarks"
  run_step "bench_steering_op" \
    "$PY" scripts/bench_steering_op.py \
      --output-dir "$OUT/micro/" --tag "$TAG"
  run_step "bench_index_building" \
    "$PY" scripts/bench_index_building.py \
      --output-dir "$OUT/micro/" --tag "$TAG"
  run_step "bench_steering_manager" \
    "$PY" scripts/bench_steering_manager.py \
      --output-dir "$OUT/micro/" --tag "$TAG"
fi

# ─── 4. Gemma-3-4B runs ──────────────────────────────────────────────────
if [[ "$DO_4B" == "1" ]]; then

  if [[ "$DO_4B_SYSTEM" == "1" ]]; then
    section "Gemma-3-4B system benchmarks"

    run_step "4B throughput_matrix" \
      "$PY" scripts/bench_throughput_matrix.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/vllm/" --tag "$TAG" \
        --batch-sizes "1,4,8,16,32" \
        --max-steering-configs 32 \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"

    run_step "4B latency" \
      "$PY" scripts/bench_latency.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/vllm/" --tag "$TAG" \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"

    run_step "4B mixed_batch" \
      "$PY" scripts/bench_mixed_batch.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/vllm/" --tag "$TAG"

    run_step "4B max_tokens sweep" \
      "$PY" scripts/bench_max_tokens.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/vllm/" --tag "$TAG"

    run_step "4B memory" \
      "$PY" scripts/bench_memory.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/vllm/" --tag "$TAG"
  fi

  if [[ "$DO_4B_ABLATION" == "1" ]]; then
    section "Gemma-3-4B ablations"

    # The single most important chart for the RFC comment:
    # eager-vs-graphs x steering-on-off.
    run_step "4B cuda_graphs" \
      "$PY" scripts/bench_cuda_graphs.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/ablation/" --tag "$TAG" \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"

    run_step "4B config_scaling" \
      "$PY" scripts/bench_config_scaling.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/ablation/" --tag "$TAG"

    run_step "4B hook_points" \
      "$PY" scripts/bench_hook_points.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/ablation/" --tag "$TAG"
  fi

  if [[ "$DO_4B_SERVING" == "1" ]]; then
    section "Gemma-3-4B online serving"

    run_step "4B serving (TTFT/TPOT/ITL/E2EL)" \
      "$PY" scripts/bench_serving.py \
        --model "$MODEL_SMALL" \
        --output-dir "$OUT/serving/" --tag "$TAG" \
        --python-bin "$PY" \
        --num-prompts 128 \
        --concurrency 16 \
        --max-tokens 256 \
        --max-steering-configs 16 \
        "${SHARED_SERVING_ARGS[@]}"
  fi
fi

# ─── 5. Gemma-3-27B runs (reduced sweeps — slower, tighter memory) ───────
if [[ "$DO_27B" == "1" ]]; then

  if [[ "$DO_27B_SYSTEM" == "1" ]]; then
    section "Gemma-3-27B system benchmarks"

    run_step "27B throughput_matrix" \
      "$PY" scripts/bench_throughput_matrix.py \
        --model "$MODEL_LARGE" \
        --output-dir "$OUT/vllm/" --tag "$TAG" \
        --batch-sizes "1,4,8,16" \
        --max-steering-configs 16 \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"

    run_step "27B latency" \
      "$PY" scripts/bench_latency.py \
        --model "$MODEL_LARGE" \
        --output-dir "$OUT/vllm/" --tag "$TAG" \
        --batch-sizes "1,4,8,16" \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"

    run_step "27B mixed_batch" \
      "$PY" scripts/bench_mixed_batch.py \
        --model "$MODEL_LARGE" \
        --output-dir "$OUT/vllm/" --tag "$TAG" \
        --batch-size 16
  fi

  if [[ "$DO_27B_ABLATION" == "1" ]]; then
    section "Gemma-3-27B ablations"

    # Only the one that matters for the RFC argument.
    run_step "27B cuda_graphs" \
      "$PY" scripts/bench_cuda_graphs.py \
        --model "$MODEL_LARGE" \
        --output-dir "$OUT/ablation/" --tag "$TAG" \
        --batch-sizes "1,4,8" \
        --warmup "$RFC_WARMUP" --iters "$RFC_ITERS"
  fi

  if [[ "$DO_27B_SERVING" == "1" ]]; then
    section "Gemma-3-27B online serving"

    run_step "27B serving (TTFT/TPOT/ITL/E2EL)" \
      "$PY" scripts/bench_serving.py \
        --model "$MODEL_LARGE" \
        --output-dir "$OUT/serving/" --tag "$TAG" \
        --python-bin "$PY" \
        --num-prompts 64 \
        --concurrency 8 \
        --max-tokens 256 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.92 \
        --max-steering-configs 16 \
        "${SHARED_SERVING_ARGS[@]}"
  fi
fi

# ─── 6. Aggregation ──────────────────────────────────────────────────────
if [[ "$DO_ANALYZE" == "1" ]]; then
  section "analysis"
  run_step "analyze" \
    "$PY" scripts/analyze.py \
      --results-dir "$OUT" \
      --output-dir "$OUT/summary" \
      --tag "$TAG"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  section "DRY RUN SUMMARY"
  printf 'Total planned steps: %d\n' "$STEP_COUNT"
  printf 'Would write to:      %s\n' "$OUT"
  printf 'Tag:                 %s\n' "$TAG"
  if [[ -n "$SHAREGPT_PATH" ]]; then
    printf 'ShareGPT workload:   %s\n' "$SHAREGPT_PATH"
  else
    printf 'ShareGPT workload:   (none — using synthetic prompts)\n'
  fi
  printf '\nRe-run without --dry-run to execute.\n'
else
  section "DONE"
  printf 'All results under: %s\n' "$OUT"
  printf 'Tag: %s\n' "$TAG"
  printf 'Next: curate the summary tables in %s/summary/ and paste into the RFC comment.\n' "$OUT"
fi
