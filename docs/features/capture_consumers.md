# Capture Consumer Benchmarks

## Scope

Benchmarks that measure the performance cost of vLLM's capture-consumer framework —
the system that extracts neural network hidden states from running inference and
delivers them to external consumers (filesystem writers, RL trainers, dashboards).

**In scope:**
- End-to-end LLM.generate() throughput overhead with consumers enabled
- CaptureManager internal latency: plan building, GPU gather (index_select), dispatch
- ActivationWriter filesystem throughput and finalize latency
- Observation-path latency (dispatch-to-consumer delivery, worker vs driver)
- Plugin work budget — how much CPU a consumer can burn per chunk before throughput drops
- Multi-hook capture cost (multiple hook types simultaneously)

**Out of scope:**
- Multi-GPU / tensor-parallel capture
- Correctness of captured activations (use existing unit tests)

---

## Scripts

### `scripts/bench_capture_e2e.py`

**Question answered:** Does enabling capture consumers measurably hurt inference throughput?

**How it works:**
- Constructs a fresh `LLM` for each consumer configuration
- Runs `warmup` generate() calls to prime CUDA graphs, then `iters` timed calls
- Reports tokens/sec and % overhead vs the no-consumer baseline
- Wall-clock timing with `torch.cuda.synchronize()` barriers

**Configurations tested:**

| Config | Description |
|---|---|
| `baseline` | No consumers |
| `logging_minimal` | 1 logging consumer, `last_prompt`, 1 layer |
| `logging_max` | 1 logging consumer, `all` positions, all layers |
| `logging_3x_same_hook` | 3 logging consumers on same hook/layer (union-gather) |
| `filesystem_minimal` | 1 filesystem consumer, `last_prompt`, 1 layer |
| `driver_minimal` | 1 driver RecordingConsumer, `last_prompt`, 1 layer |

**Key files:**
- `src/steering_bench/capture_consumers/consumers.py` — `RecordingDriverConsumer`
- `src/steering_bench/capture_consumers/runner.py` — `make_prompts`, `get_model_config`
- vLLM: `vllm/entrypoints/llm.py:256` — `capture_consumers` parameter
- vLLM: `vllm/v1/capture/consumers/logging.py` — LoggingConsumer
- vLLM: `vllm/v1/capture/consumers/filesystem/consumer.py` — FilesystemConsumer

**Output:** `results/capture/capture_e2e_*.json` (one file per timed config×batch_size)

---

### `scripts/bench_capture_manager.py`

**Question answered:** Which part of the manager hot path is expensive?

**Three phases timed separately:**

1. **build_ms** — `CaptureManager.build_step_plan()`: pure CPU, O(requests × consumers ×
   layers × positions). Expands position selectors and builds CUDA gather-index tensors.
2. **hook_ms** — `CaptureManager.on_hook()` × layers: GPU `index_select` that copies
   desired rows from `hidden_states` into scratch buffers.
3. **dispatch_ms** — `CaptureManager.dispatch_step_captures()`: CPU fan-out that slices
   scratch tensors (GPU→CPU copy) and calls `submit_chunk` on each sink.

**How it works:**
- Constructs `CaptureManager` directly on CUDA with `NullCaptureSink`s (no I/O)
- Simulates a prefill step: all `prompt_len` tokens scheduled in one step
- CUDA events for `hook_ms`; `time.perf_counter` for `build_ms` and `dispatch_ms`
- Registers and finalizes requests each iteration to reset internal state

**Sweep matrix:**

| Dimension | Values |
|---|---|
| `batch_size` | 1, 8, 32 |
| `num_consumers` | 1, 2, 4, 8 |
| `position_type` | last_prompt, all_prompt |
| `num_layers` | 1, 6, 12 |

**Key files:**
- `src/steering_bench/capture_consumers/consumers.py` — `NullCaptureSink`
- vLLM: `vllm/v1/capture/manager.py` — `CaptureManager.__init__`, `build_step_plan`,
  `on_hook`, `dispatch_step_captures`
- vLLM: `vllm/v1/capture/plan.py` — `CaptureBatchView`

**Output:** `results/capture/capture_manager_*.json` (one file with full sweep)

---

### `scripts/bench_capture_filesystem.py`

**Question answered:** Can the ActivationWriter keep up with the model?

**How it works:**
- Drives `ActivationWriter` directly (no LLM, no CaptureManager)
- Submits `num_requests × steps_per_request` WriteTask entries + `num_requests`
  FinalizeTasks in one pass, waits for all FinalizeTask completions
- Uses `writer.add_status_callback()` to record exact completion timestamps
- Computes sustained throughput (MB/s) and finalize latency distribution

**Sweep matrix:**

| Dimension | Values |
|---|---|
| `writer_threads` | 1, 2, 4, 8 |
| `hidden_size` | 768 (opt-125m), 4096 (7B), 8192 (70B) |
| `num_requests` | 32, 128 |
| `steps_per_request` | 8, 32 |

**Key files:**
- vLLM: `vllm/v1/capture/consumers/filesystem/writer.py` — `ActivationWriter`,
  `WriteTask`, `FinalizeTask`

**Output:** `results/capture/capture_filesystem_*.json` (one file with full sweep)

---

### `scripts/bench_capture_latency.py`

**Question answered:** For an online plugin, how long between the capture system
having an activation and my consumer's callback firing?

Two modes controlled by `--mode`:

#### `--mode microbench` — dispatch-added delivery latency

Drives `CaptureManager` directly with `TimestampingSink`.  Records the
interval between `dispatch_step_captures()` being called and each
`submit_chunk` firing.  This is the cost the capture system adds *on top
of* the forward pass.  Results are reported in **microseconds** (`*_us`).

The microbench deliberately excludes forward-pass kernels that would
run between `on_hook` and `dispatch` during real inference; user-facing
latency is that forward time plus the dispatch number reported here.

Sweep: `batch_size × num_consumers × num_layers × position_type`.
Reports per-consumer-idx p50/p99/p99.9 plus an aggregated "all" view.

#### `--mode e2e` — end-to-end on_capture latency, worker vs driver

Runs real `LLM.generate()` with `TimestampingConsumer` and sweeps
`location ∈ {worker, driver}`.  Records interval from `t_request_submit`
(just before `generate()`) to each `on_capture` firing.  Results in
**milliseconds** (`*_ms`).

The `location` dimension is the headline: driver-side consumers go
through the worker→driver IPC bridge (multiprocess queue + thread hop
+ batched adapter) while worker-side stays in-process.  Most reward
consumers run driver-side, so this is the number they actually
observe.

Sweep: `location × batch_size × num_layers × position_type`.

**Key files:**
- `src/steering_bench/capture_consumers/consumers.py` —
  `TimestampingSink`, `TimestampingConsumer`
- vLLM: `vllm/v1/capture/manager.py` — `dispatch_step_captures`
- vLLM: `vllm/v1/capture/consumer.py` — `_BatchedAdapter` (driver path)

**Output:** `results/capture/capture_latency_microbench_*.json` or
`results/capture/capture_latency_e2e_*.json`.

---

### `scripts/bench_capture_plugin_work.py`

**Question answered:** How much CPU can my plugin spend per chunk
before it starts hurting throughput?

Two modes.  Both drive a `SimulatedWorkSink` (microbench) or
`SimulatedWorkConsumer` (e2e) with a configurable `work_us` per call
and one of three work modes:

| Mode | Behavior | Realistic for |
|---|---|---|
| `busy` | Spin-wait on `perf_counter_ns`.  Accurate to sub-μs. | Synchronous CPU-bound plugins (inline scoring, norms). |
| `sleep` | `time.sleep(work_us / 1e6)`.  Skipped for `work_us < 100` because scheduler resolution floors at ~50 μs. | Yielding work; coarse model. |
| `queue` | Enqueue a sentinel; worker thread drains + sleeps.  `submit_chunk` blocks only when the queue fills. | The realistic consumer pattern — filesystem consumer, streaming trainers. Captures backpressure. |

#### `--mode microbench`

Sweep: `work_us × work_modes × num_consumers`.  Measures `dispatch_ms`
per config.  The plan deliberately does **not** subtract a spurious
"pipeline overhead" — just reports the `(work_us, dispatch_ms)` series
per `(mode, num_consumers)`.  Slope = per-chunk cost; intercept = fixed
overhead.

#### `--mode e2e`

Sweep: `(work_us, mode) × batch_size` on real `LLM.generate()`.
Reports `tokens_per_sec` and `overhead_pct` vs a no-consumer baseline
collected per batch size.  Emits a `budgets` section in the result
JSON: the largest `work_us` per `(mode, batch_size)` at which overhead
stays under `--budget-threshold-pct` (default 5%).  This is the single
headline number for consumer authors: *"at batch=8 with a queueing
consumer, you can spend up to N μs per chunk before throughput drops
5%."*

**Key files:**
- `src/steering_bench/capture_consumers/consumers.py` —
  `SimulatedWorkSink`, `SimulatedWorkConsumer`, `_WorkSimulator`

**Output:** `results/capture/capture_plugin_work_microbench_*.json` or
`results/capture/capture_plugin_work_e2e_*.json`.

---

### `scripts/bench_capture_manager.py --hook-sets ...`

The existing manager microbenchmark gained a `--hook-sets` flag for
multi-hook capture.  Pass a semicolon-separated list of
comma-separated hook names:

```
--hook-sets "post_mlp;post_mlp,post_attn;post_mlp,post_attn,pre_mlp,pre_attn"
```

Each set becomes one sweep run, adding a `num_hooks` dimension to the
output JSON and table.  Default behavior (unset) is backward-compatible
with the original single-hook sweep via `--hook-name`.

**Expected finding:** `build_ms` and `hook_ms` should scale near-linearly
in `num_hooks × num_layers`.  If they don't, that's a real finding —
surface it in the summary table.

---

## Shared Helpers

### `src/steering_bench/capture_consumers/consumers.py`

- **`NullCaptureSink`** — Implements the `CaptureSink` protocol, discards all data,
  marks results `ok` immediately on finalize. Used by `bench_capture_manager.py` to
  isolate manager overhead from I/O.
- **`RecordingDriverConsumer`** — `CaptureConsumer` subclass with `location="driver"`.
  Increments a counter per `on_capture` call (no tensor retention).  Used by
  `bench_capture_e2e.py` to verify the driver consumer path is live.
- **`TimestampingSink`** — `CaptureSink` that records `perf_counter_ns()` inside
  `submit_chunk`.  Exposes `drain_timestamps()`.  Used by the latency microbench.
- **`TimestampingConsumer`** — `CaptureConsumer` that records `perf_counter_ns()` in
  `on_capture`.  `location` is configurable per-instance so one class serves both
  worker and driver sweeps.  Used by the latency E2E benchmark.
- **`SimulatedWorkSink`** — `CaptureSink` that spends `work_us` per `submit_chunk`
  using one of three modes (`busy` / `sleep` / `queue`).  Used by the plugin-work
  microbench.
- **`SimulatedWorkConsumer`** — `CaptureConsumer` counterpart of `SimulatedWorkSink`;
  work runs inside `on_capture`.  Used by the plugin-work E2E benchmark.
- **`_WorkSimulator`** — internal helper shared by the two simulated-work classes.
  Encapsulates the busy/sleep/queue logic plus a worker thread for queue mode.

### `src/steering_bench/capture_consumers/runner.py`

- **`MODEL_CONFIGS`** — hidden_size and num_layers for common models
- **`get_model_config(model)`** — lookup with fallback
- **`make_llm(...)`** — thin wrapper around `LLM()` with capture_consumers forwarded
- **`make_prompts(num_prompts, prompt_len)`** — dummy prompt generator

---

## Running

```bash
cd /home/nymph/Code/steering-bench
VLLM_PY=/home/nymph/Code/vllm/capture-consumers/.venv/bin/python

# No model needed (~2-5 min depending on sweep size).  Use --bench-dir for real disk.
$VLLM_PY scripts/bench_capture_filesystem.py --bench-dir ~/bench_writer_tmp

# Storage path (~2 min, requires GPU, no model load)
$VLLM_PY scripts/bench_capture_manager.py

# Multi-hook sweep (~5 min)
$VLLM_PY scripts/bench_capture_manager.py \
  --hook-sets "post_mlp;post_mlp,post_attn;post_mlp,post_attn,pre_mlp,pre_attn"

# E2E overhead across consumer configs (~10-15 min, loads model 6 times)
$VLLM_PY scripts/bench_capture_e2e.py

# Observation-path latency
$VLLM_PY scripts/bench_capture_latency.py --mode microbench     # ~2 min
$VLLM_PY scripts/bench_capture_latency.py --mode e2e            # ~15 min

# Plugin work budget
$VLLM_PY scripts/bench_capture_plugin_work.py --mode microbench # ~3 min
$VLLM_PY scripts/bench_capture_plugin_work.py --mode e2e        # ~20 min

# Quick smoke test: baseline + logging_minimal, batch=1 only
$VLLM_PY scripts/bench_capture_e2e.py \
  --configs baseline,logging_minimal \
  --batch-sizes 1 \
  --warmup 1 --iters 2
```

---

## Invariants

- **Consumer isolation is assumed** — benchmarks do not test error paths, only the hot
  path where all consumers succeed.
- **Single GPU, single process** — no tensor-parallel setup. On multi-GPU hosts the
  benchmark uses GPU 0 by default.
- **NullCaptureSink finalizes synchronously** — `wait_for_result()` returns immediately,
  so finalize timeout in `finalize_request()` never fires during the manager benchmark.
- **Filesystem benchmark uses tmpdir** — results are not persisted across runs. Use
  `--output-dir` on the script itself only for JSON metric output, not capture data.
