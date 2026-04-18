# Capture Consumer Benchmarks

## Scope

Benchmarks that measure the performance cost of vLLM's capture-consumer framework —
the system that extracts neural network hidden states from running inference and
delivers them to external consumers (filesystem writers, RL trainers, dashboards).

**In scope:**
- End-to-end LLM.generate() throughput overhead with consumers enabled
- CaptureManager internal latency: plan building, GPU gather (index_select), dispatch
- ActivationWriter filesystem throughput and finalize latency

**Out of scope:**
- Driver-consumer IPC throughput (torch.multiprocessing queue)
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

## Shared Helpers

### `src/steering_bench/capture_consumers/consumers.py`

- **`NullCaptureSink`** — Implements the `CaptureSink` protocol, discards all data,
  marks results `ok` immediately on finalize. Used by `bench_capture_manager.py` to
  isolate manager overhead from I/O.
- **`RecordingDriverConsumer`** — `CaptureConsumer` subclass with `location="driver"`.
  Accumulates `(key, tensor)` pairs in memory. Used by `bench_capture_e2e.py` to
  verify the driver consumer path is live.

### `src/steering_bench/capture_consumers/runner.py`

- **`MODEL_CONFIGS`** — hidden_size and num_layers for common models
- **`get_model_config(model)`** — lookup with fallback
- **`make_llm(...)`** — thin wrapper around `LLM()` with capture_consumers forwarded
- **`make_prompts(num_prompts, prompt_len)`** — dummy prompt generator

---

## Running

```bash
cd /home/nymph/Code/steering-bench

# No model needed (~2–5 min depending on sweep size)
uv run scripts/bench_capture_filesystem.py

# Requires GPU, no model load (~2 min)
uv run scripts/bench_capture_manager.py

# Requires GPU + model download (~10–15 min, loads model 6 times)
uv run scripts/bench_capture_e2e.py

# Quick smoke test: baseline + logging_minimal, batch=1 only
uv run scripts/bench_capture_e2e.py \
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
