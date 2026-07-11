# Capture-consumer profiling harness

Standalone tools for profiling the vLLM steering fork's **capture-consumer**
pipeline overhead (the `LLM(capture_consumers=[...])` path). Deliberately
vllm-direct — they import `vllm.LLM`, not `steering_bench` — so the profiler
sits as close to the code under test as possible and they run on the fork's own
(possibly older-Python) venv.

## Scripts

| file | role |
|---|---|
| `profile_capture.py` | Loads a model, runs a warmup + timed `generate()` loop with a chosen capture config, reports tok/s. `--mode torch` brackets the timed region with vLLM's built-in **worker** torch profiler (via `ProfilerConfig`), the only way to see the GPU work that runs in the V1 EngineCore subprocess. An NVTX range (`timed_region`) is emitted for nsys scoping. |
| `analyze_trace.py` | Aggregates torch-profiler chrome traces (gzip-aware) by op name and **diffs** two runs (baseline vs capture) to surface exactly which GPU-kernel / memcpy / cpu-op the capture path adds. |
| `agg_folded.py` | Aggregates a py-spy folded (`--format raw`) stack file: total samples, capture-inclusive share, top self-time frames — for the CPU-side flamegraph view. |
| `run_profiles.sh` | Orchestrates the three profilers (`torch` \| `nsys` \| `pyspy`). Paths are env-overridable (`PYTHON`, `NSYS`, `PYSPY`). |

## Capture configs (`--config`)

`baseline` (no consumers), `logging_minimal`, `logging_layers` (all layers ×
last position), `logging_positions` (one layer × all positions), `logging_max`
(all layers × all positions), `logging_max_silent` (as `logging_max` but at a
log level below the threshold, so `logger.log` short-circuits — isolates the
capture *pipeline* cost from log I/O).

## Usage

```bash
export PYTHON=/path/to/fork-venv/bin/python          # vllm-direct venv
export HF_HUB_OFFLINE=1                               # if the model is cached
# throughput / overhead of a single config
$PYTHON scripts/profiling/profile_capture.py --config logging_max --batch-size 16

# torch op-level diff (writes traces, then diffs)
$PYTHON scripts/profiling/profile_capture.py --config baseline    --mode torch --trace-dir trace_baseline
$PYTHON scripts/profiling/profile_capture.py --config logging_max --mode torch --trace-dir trace_capture
$PYTHON scripts/profiling/analyze_trace.py baseline=trace_baseline capture=trace_capture

# CPU flamegraph (py-spy follows the EngineCore subprocess)
py-spy record --subprocesses --format raw -o stacks.folded -- \
  $PYTHON scripts/profiling/profile_capture.py --config logging_max --iters 40
$PYTHON scripts/profiling/agg_folded.py stacks.folded
```

## Notes

- **nsys** traces the driver process but **not** the V1 EngineCore subprocess's
  CUDA (CUPTI does not attach across the spawn); use it for NVTX wall timing and
  rely on the built-in torch profiler for the worker's GPU work.
- **ncu** needs GPU performance-counter access (`NVreg_RestrictProfilingToAdminUsers=0`
  or `sudo`); it is only relevant when the bottleneck is a GPU kernel.
- The harness force-exits after reporting; it shuts the engine core down first
  so the subprocess releases VRAM between sequential configs.
