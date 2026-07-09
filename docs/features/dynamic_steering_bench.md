# dynamic_steering benchmark

## Scope

Measures the end-to-end wall-clock `generate()` throughput overhead of vLLM's
**dynamic steering** and **capture-consumer** tiers, relative to a baseline with
neither capture nor steering active, on a real **gemma4** model. Doubles as a
performance benchmark for the capture-consumer feature: it puts the preexisting
**async** path next to the new **sync** and in-graph **dynamic** (same-token)
tiers.

### Non-scope

- Correctness of steering (covered by the vLLM-side e2e tests).
- Operator-set steering on non-gemma4 models (`bench_steering_with_capture.py`).
- Microbenchmarks of the kernel/manager in isolation (`bench_steering_op.py`,
  `bench_steering_manager.py`).

## Why gemma4 only

Every capture consumer — and therefore all dynamic steering, which is driven by
a consumer reading the residual — reads from the `maybe_capture_residual` tap.
In the dynamic-steering branch that tap exists **only in
`vllm/model_executor/models/gemma4.py`**. Steering *hooks* (`apply_layer_steering`)
exist in many models, but consumer-driven dynamic steering does not, so the
benchmark is constrained to a gemma4 model
(`~/Models/gemma-4-31B-it-Q4_K_S.gguf`).

## Arms

| arm | enable_steering | consumer | what it adds vs the row above |
|---|---|---|---|
| `off` | no | none | baseline (cold path) |
| `capture_async` | no | async `on_capture` reader | capture gather/dispatch pipeline |
| `capture_sync` | no | sync `on_step` reader | per-step `on_step` vs finalize `on_capture` |
| `steer_async` | yes | async tier via action queue | steering kernel + async transport |
| `steer_sync` | yes | sync tier via `on_step` | sync transport (exactly-1-step) |
| `steer_dynamic` | yes | in-graph monitor gating the tier | same-token monitor op cost |

The steering arms keep a **global** decode tier active for the whole batch from
the first decode step on, so every decode token is steered every step
(steady-state overhead); the per-step decision work is a single fixed probe so
the comparison isolates the *transport*, not policy. `steer_dynamic` uses a
saturated negative monitor threshold so the gate stays ~1 (steering live) — it
measures the in-graph monitor's cost, not a disengaged probe.

## Data / control flow

```
bench_dynamic_steering.py (parent)
  for (arm, batch_size):
    subprocess: bench_dynamic_steering.py --cell --arm A --batch-size N
      → VllmSteeringEngine.configure_capture([CaptureConsumerSpec(bench_A, ...)])
      → engine.load(model, SteeringConfig(enable_steering=...), **fork_opts)
      → warmup × W, activation gate via engine.capture_status() /
        engine.live_capture_consumers(), then timed engine.generate() × I
      → print "CELL_RESULT {json}"
    parent parses the line, computes overhead_pct vs the `off` cell,
    write_result(benchmark="steering.dynamic", ...) → results/dynamic_steering/
  print summary table
```

Each cell runs in a **fresh subprocess** because the dynamic-steering action
queue is process-global and vLLM's residual weight memory does not free cleanly
across `LLM` instances — a subprocess per cell keeps state and memory clean.

**Phase 4 seam usage (partial migration):** consumer declaration is the typed
`CaptureConsumerSpec`, and LLM construction + activation-status introspection go
through the CaptureEngine seam (`configure_capture` / `capture_status` /
`live_capture_consumers`) instead of raw `from vllm import LLM` +
`collective_rpc`. The subprocess-per-cell orchestration and fork-only load knobs
(`max_dynamic_steering_configs`, `enable_row_monitor`) stay vLLM-specific;
`enable_prefix_caching` / `max_steering_configs` follow the `SteeringConfig`
defaults (True / 4), which match vLLM's defaults.

## Files

- `scripts/bench_dynamic_steering.py` — runner (parent fan-out + `--cell` mode).
- `src/steering_bench/capture_consumers/bench_consumers.py` — the six arms'
  consumers (`Bench{Capture,Steer}{Async,Sync}`, `BenchSteerDynamic`),
  registered as `vllm.capture_consumers` entry points in `pyproject.toml`.
- `src/steering_bench/timing.py`, `output.py` — shared stats + JSON schema.

## Invariants / constraints

- Worker-side consumers must be resolvable by name (entry points) — instances
  can't be passed for `location="worker"`.
- Requires `VLLM_USE_FLASHINFER_SAMPLER=0` on the 3090 nodes (CUB JIT failure
  unrelated to steering) and the in-process engine core
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) to avoid CUDA-reinit-after-fork.
- Runs against the dynamic-steering vLLM branch (node0 `~/Code/vllm-fork`).

## Usage

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python scripts/bench_dynamic_steering.py \
  --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf --layer 30 \
  --batch-sizes 1,8,32 --output-len 64 --prompt-len 64
```

`--arms off,steer_dynamic` runs a subset; `--enforce-eager` disables cudagraphs.
