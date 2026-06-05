#!/bin/bash
# H100 launcher for environments WITHOUT clock-pinning permission (Vast.ai, etc.).
#
# Compensates for thermal drift with:
#   - 5 trials per cell (vs 3) — better variance estimation
#   - Interleaved trial order — thermal trajectory affects all variants equally
#   - 90s cooldown between worktree swaps — card cools before next cell
#   - Per-cell clock logging — analyze.py can filter contaminated cells post-hoc
#   - Shorter modes lists per cell — each cell fits in one thermal burst
#
# This is the "best you can do without root" plan.  Cross-cell deltas of <5ms
# should still be treated with suspicion; >10ms deltas that survive 5-trial
# averaging are likely real.
#
# Pre-run: try `nvidia-smi -pm 1` (sometimes works without sudo) and/or
# `nvidia-smi -pl <W>` (a lower power cap forces a stable lower clock band).
#
# Overrides via env: BENCH_ROOT, VLLM_WORKTREE.

set -u

BENCH_ROOT="${BENCH_ROOT:-/workspace/steering-bench}"
VLLM_WORKTREE="${VLLM_WORKTREE:-/workspace/vllm/steering}"
COOLDOWN="${COOLDOWN:-90}"
TRIALS="${TRIALS:-5}"

OUT_ROOT="$BENCH_ROOT/results/h100-unpinned"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

PY="$BENCH_ROOT/.venv/bin/python"
export PYTHONPATH="$VLLM_WORKTREE"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME (unpinned)"
echo "Trials/cell: $TRIALS, cooldown between cells: ${COOLDOWN}s"
echo ""

# Try persistence-mode (sometimes allowed without sudo on Vast)
nvidia-smi -pm 1 2>&1 | head -1 || true

cd "$BENCH_ROOT"
echo "=== launcher start: $(date -Iseconds) ===" | tee "$LOG_DIR/launcher.log"

cooldown() {
  echo "  cooldown ${COOLDOWN}s (current clock: $(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits) MHz, temp: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)C)"
  sleep "$COOLDOWN"
}

# ----------------------------------------------------------------------
# 1. Serving — run each mode in its own server, multiple trials, interleaved.
#    Per-mode launches kill the (worktree, mode, trial) cell down to ~2 min
#    each so thermal drift within a cell is bounded.
# ----------------------------------------------------------------------
echo "=== [1/3] bench_serving (one-mode-per-launch, ${TRIALS} trials) ===" | tee -a "$LOG_DIR/launcher.log"

# Modes ordered by expected duration — shortest first so cooldown matters less
MODES="enabled_idle named_shared all_steered_shared per_request_n4 per_request_n16"

for trial in $(seq 1 "$TRIALS"); do
  for mode in $MODES; do
    tag="h100-unpinned-${mode}-t${trial}"
    echo "  $(date -Iseconds)  ${tag}" | tee -a "$LOG_DIR/launcher.log"
    pkill -f "api_server.*--port 8765" 2>/dev/null
    sleep 3

    "$PY" "$BENCH_ROOT/scripts/bench_serving.py" \
      --model google/gemma-3-4b-it \
      --num-prompts 128 --concurrency 8 \
      --max-tokens 128 --prompt-len 256 \
      --max-model-len 2048 --gpu-memory-utilization 0.9 \
      --max-steering-configs 16 \
      --modes "$mode" \
      --tag "$tag" --python-bin "$PY" \
      --output-dir "$OUT_ROOT/serving/" \
      >"$LOG_DIR/serving-${tag}.log" 2>&1

    pkill -f "api_server.*--port 8765" 2>/dev/null
    cooldown
  done
done

# ----------------------------------------------------------------------
# 2. Throughput matrix — already does its own warmup loops, but bump iters
#    so each cell aggregates over more samples (smoothing thermal drift).
# ----------------------------------------------------------------------
echo "=== [2/3] bench_throughput_matrix (3 trials) ===" | tee -a "$LOG_DIR/launcher.log"
for trial in 1 2 3; do
  echo "  $(date -Iseconds)  throughput-matrix t${trial}" | tee -a "$LOG_DIR/launcher.log"
  "$PY" "$BENCH_ROOT/scripts/bench_throughput_matrix.py" \
    --model google/gemma-3-4b-it \
    --warmup 5 --iters 30 \
    --max-tokens 128 --prompt-len 64 \
    --batch-sizes "1,4,8,16,32" \
    --fractions "0.0,0.25,0.5,0.75,1.0" \
    --max-steering-configs 4 \
    --tag "h100-unpinned-t${trial}" \
    --output-dir "$OUT_ROOT/throughput-matrix/" \
    >"$LOG_DIR/throughput-matrix-t${trial}.log" 2>&1
  cooldown
done

# ----------------------------------------------------------------------
# 3. Modes matrix — headline preset, single run (this script's grid is
#    already a 60+ cell sweep; running 3 trials of it would take 6+ hours).
# ----------------------------------------------------------------------
echo "=== [3/3] bench_steering_modes_matrix --preset headline ===" | tee -a "$LOG_DIR/launcher.log"
"$PY" "$BENCH_ROOT/scripts/bench_steering_modes_matrix.py" \
  --model google/gemma-3-4b-it \
  --preset headline \
  --max-tokens 128 --warmup 3 --iters 10 \
  --tag h100-unpinned \
  --output-dir "$OUT_ROOT/modes-matrix/" \
  2>&1 | tee "$LOG_DIR/modes-matrix.log"

# ----------------------------------------------------------------------
# Summary + clock-drift sanity check
# ----------------------------------------------------------------------
echo "=== launcher done: $(date -Iseconds) ===" | tee -a "$LOG_DIR/launcher.log"
echo ""
echo "Results:"
echo "  $OUT_ROOT/serving/            ($(ls $OUT_ROOT/serving/ 2>/dev/null | wc -l) JSONs)"
echo "  $OUT_ROOT/throughput-matrix/  ($(ls $OUT_ROOT/throughput-matrix/ 2>/dev/null | wc -l) JSONs)"
echo "  $OUT_ROOT/modes-matrix/       ($(ls $OUT_ROOT/modes-matrix/ 2>/dev/null | wc -l) JSONs)"
echo ""
echo "Clock drift across serving cells (look for outliers):"
OUT_ROOT="$OUT_ROOT" "$PY" -c '
import json, glob, statistics, os
root = os.environ["OUT_ROOT"] + "/serving"
clocks = []
for f in sorted(glob.glob(f"{root}/*.json")):
    d = json.load(open(f))
    clk = d.get("environment", {}).get("gpu_clock_current_mhz")
    if clk is not None:
        clocks.append(clk)
if clocks:
    mn, md, mx = min(clocks), int(statistics.median(clocks)), max(clocks)
    print(f"  cells: {len(clocks)}, clock min/median/max = {mn}/{md}/{mx} MHz")
    print(f"  drift: {mx-mn} MHz ({100*(mx-mn)/md:.1f}% of median)")
    if mx - mn > 200:
        print(f"  WARNING: large drift — filter cells whose clock differs from median by >5%")
'
