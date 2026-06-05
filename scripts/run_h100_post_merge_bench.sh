#!/bin/bash
# Post-merge re-bench launcher — runs the three bench suites on the
# current feat/steering state (binary wire #162 + per-layer scales #163 +
# prematerialize #161 + async-h2d #158 + remove-auto-promote #164 +
# jit-prewarm #157).
#
# Designed to be copy-pasted onto any single-GPU box with sudo. Works
# unchanged on A100, H100, H200, GH200, RTX 6000 Pro, etc. Override
# paths via env:
#   BENCH_ROOT      — steering-bench checkout (default /home/nymph/Code/steering-bench)
#   VLLM_WORKTREE   — vllm feat/steering worktree (default /home/nymph/Code/vllm/steering)
#   GPU_CLOCK_MHZ   — pin target in MHz (default 1500). Run
#                     probe_sustainable_clock.sh first to find the right value
#                     for your specific card / cooling. Aliased from the older
#                     H100_GR_CLOCK env var for backwards compat.
#   GPU_TAG         — output-dir / tag label (default 'post-merge'). Set to
#                     'a100-post-merge', 'h100-post-merge', etc. so the
#                     results dir + tag say which GPU was benched.
#   MODES_PRESET    — bench_steering_modes_matrix preset. Default 'full'
#                     (540 cells, ~6-9 hr on H100-class). Set to 'mid'
#                     (160 cells, ~2-3 hr) or 'headline' (24 cells, ~25 min)
#                     for faster runs.
#
# Approximate sustainable clocks (pin with margin below these):
#   A100 40/80GB    ~1200-1300 MHz
#   H100 SXM5       ~1500-1700 MHz
#   H100 PCIe       ~1400-1500 MHz
#   H200            ~1500-1700 MHz
#   GH200           ~1500-1700 MHz
#   RTX 6000 Pro    ~1700-1900 MHz
# Run scripts/probe_sustainable_clock.sh to get an empirical value.
#
# Pin GPU clocks BEFORE running:
#   sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc $GPU_CLOCK_MHZ

set -u

BENCH_ROOT="${BENCH_ROOT:-/home/nymph/Code/steering-bench}"
VLLM_WORKTREE="${VLLM_WORKTREE:-/home/nymph/Code/vllm/steering}"
# Backwards-compat: accept the older H100_GR_CLOCK name as fallback.
GPU_CLOCK_MHZ="${GPU_CLOCK_MHZ:-${H100_GR_CLOCK:-1500}}"
GPU_TAG="${GPU_TAG:-post-merge}"

OUT_ROOT="$BENCH_ROOT/results/${GPU_TAG}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="$BENCH_ROOT/.venv/bin/python"
export PYTHONPATH="$VLLM_WORKTREE"

ACTUAL=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
echo "Current clock: ${ACTUAL} MHz (target ~${GPU_CLOCK_MHZ} MHz)"
DELTA=$(( ACTUAL > GPU_CLOCK_MHZ ? ACTUAL - GPU_CLOCK_MHZ : GPU_CLOCK_MHZ - ACTUAL ))
if [ "$DELTA" -gt 50 ]; then
  echo "WARNING: clock not pinned within 50 MHz of ${GPU_CLOCK_MHZ}."
  echo "Run: sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc ${GPU_CLOCK_MHZ}"
  echo "Continuing — analysis will flag drift via gpu_clock_current_mhz."
fi

echo "=== launcher start: $(date -Iseconds) ===" | tee "$LOG_DIR/launcher.log"

# Run from BENCH_ROOT so relative --python-bin etc. resolve.
cd "$BENCH_ROOT"

# ----------------------------------------------------------------------
# 1. Throughput matrix — mode × batch_size × inline-fraction
# ----------------------------------------------------------------------
echo "=== [1/3] bench_throughput_matrix.py ===" | tee -a "$LOG_DIR/launcher.log"
date -Iseconds | tee -a "$LOG_DIR/launcher.log"

"$PY" "$BENCH_ROOT/scripts/bench_throughput_matrix.py" \
  --model google/gemma-3-4b-it \
  --warmup 5 \
  --iters 20 \
  --max-tokens 128 \
  --prompt-len 64 \
  --batch-sizes "1,4,8,16,32" \
  --fractions "0.0,0.25,0.5,0.75,1.0" \
  --max-steering-configs 4 \
  --tag "${GPU_TAG}" \
  --output-dir "$OUT_ROOT/throughput-matrix/" \
  2>&1 | tee "$LOG_DIR/throughput-matrix.log"

# ----------------------------------------------------------------------
# 2. Steering-modes matrix — full preset (540 cells, ~6-9 hr on H100-class).
#    Override with MODES_PRESET=headline (24 cells, ~25 min) or
#    MODES_PRESET=mid (160 cells, ~2-3 hr) if shorter run needed.
# ----------------------------------------------------------------------
MODES_PRESET="${MODES_PRESET:-full}"
echo "=== [2/3] bench_steering_modes_matrix.py --preset ${MODES_PRESET} ===" | tee -a "$LOG_DIR/launcher.log"
date -Iseconds | tee -a "$LOG_DIR/launcher.log"

"$PY" "$BENCH_ROOT/scripts/bench_steering_modes_matrix.py" \
  --model google/gemma-3-4b-it \
  --preset "${MODES_PRESET}" \
  --max-tokens 128 \
  --warmup 3 \
  --iters 5 \
  --tag "${GPU_TAG}" \
  --output-dir "$OUT_ROOT/modes-matrix/" \
  2>&1 | tee "$LOG_DIR/modes-matrix.log"

# ----------------------------------------------------------------------
# 3. Serving — same params as the pinned-3090 runs for direct comparison
#    (n=128, conc=8, packed wire format, drained warmup)
# ----------------------------------------------------------------------
echo "=== [3/3] bench_serving.py (n=128, conc=8, packed wire format) ===" | tee -a "$LOG_DIR/launcher.log"
date -Iseconds | tee -a "$LOG_DIR/launcher.log"

pkill -f "api_server.*--port 8765" 2>/dev/null
sleep 3

"$PY" "$BENCH_ROOT/scripts/bench_serving.py" \
  --model google/gemma-3-4b-it \
  --num-prompts 128 \
  --concurrency 8 \
  --max-tokens 128 \
  --prompt-len 256 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --max-steering-configs 16 \
  --modes "enabled_idle,named_shared,all_steered_shared,per_request_n4,per_request_n16" \
  --tag "${GPU_TAG}" \
  --python-bin "$PY" \
  --output-dir "$OUT_ROOT/serving/" \
  2>&1 | tee "$LOG_DIR/serving.log"

pkill -f "api_server.*--port 8765" 2>/dev/null

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo "=== launcher done: $(date -Iseconds) ===" | tee -a "$LOG_DIR/launcher.log"
echo "Results:"
echo "  $OUT_ROOT/throughput-matrix/  ($(ls $OUT_ROOT/throughput-matrix/ 2>/dev/null | wc -l) JSONs)"
echo "  $OUT_ROOT/modes-matrix/       ($(ls $OUT_ROOT/modes-matrix/ 2>/dev/null | wc -l) JSONs)"
echo "  $OUT_ROOT/serving/            ($(ls $OUT_ROOT/serving/ 2>/dev/null | wc -l) JSONs)"
echo "Logs at $LOG_DIR/"
echo "Final clock: $(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits) MHz"
