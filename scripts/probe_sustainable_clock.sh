#!/bin/bash
# Probe sustainable GPU clock for the bench.
#
# Runs a sustained ~3 minute compute workload, samples clocks.current.graphics
# every second, and reports:
#   - clock at t=10s   (cold; will be near boost)
#   - clock at t=60s   (thermal ramp underway)
#   - clock at t=120s  (close to steady-state)
#   - clock at t=180s  (steady-state)
#   - the throttle reasons that fired
#
# Pick a pin value at or slightly below the t=180s reading. That keeps every
# bench cell at the same clock regardless of when in the sweep it ran, so
# cross-cell deltas reflect code, not thermal headroom.
#
# Usage:
#   bash probe_sustainable_clock.sh

set -u

VENV=${VENV:-/home/nymph/Code/steering-bench/.venv}
LOG=/tmp/clock_probe.log
> "$LOG"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
BOOST_MAX=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU_NAME"
echo "Boost (max) clock: ${BOOST_MAX} MHz"
echo "Logging clock samples to $LOG (every 1s for 180s)"
echo ""

# Background sampler: dump current clock + throttle reasons every second
(
  for i in $(seq 1 180); do
    ts=$(date +%s)
    clk=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1)
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    reasons=$(nvidia-smi --query-gpu=clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader)
    echo "$i $clk $temp $power $reasons" >> "$LOG"
    sleep 1
  done
) &
SAMPLER_PID=$!

# Sustained load: matrix multiply loop. Sized to ~70-80% util on H100/3090.
echo "Starting 180s sustained matmul load..."
"$VENV/bin/python" - <<'PY' &
import torch, time
torch.set_grad_enabled(False)
device = torch.device("cuda:0")
N = 4096
a = torch.randn(N, N, device=device, dtype=torch.float16)
b = torch.randn(N, N, device=device, dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 180:
    for _ in range(100):
        a = torch.matmul(a, b)
    torch.cuda.synchronize()
PY
LOAD_PID=$!

wait $LOAD_PID
kill $SAMPLER_PID 2>/dev/null
wait $SAMPLER_PID 2>/dev/null

echo ""
echo "=== Clock samples (t / clock_MHz / temp_C / power_W / throttle reasons) ==="
for t in 10 30 60 90 120 150 180; do
    line=$(awk -v t="$t" '$1==t' "$LOG")
    echo "t=${t}s: $line"
done

echo ""
echo "=== Summary ==="
MIN_CLK=$(awk 'NR>30 {print $2}' "$LOG" | sort -n | head -1)
SS_CLK=$(tail -30 "$LOG" | awk '{print $2}' | sort -n | head -1)
MAX_TEMP=$(awk '{print $3}' "$LOG" | sort -n | tail -1)
echo "Min clock observed (after t=30s):     ${MIN_CLK} MHz"
echo "Min clock in last 30s (steady-state): ${SS_CLK} MHz"
echo "Max temperature:                      ${MAX_TEMP} C"
echo ""
echo "Recommended -lgc value: ${SS_CLK} MHz (or 50 MHz below for safety margin)"
echo "Pin with: sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc ${SS_CLK}"
