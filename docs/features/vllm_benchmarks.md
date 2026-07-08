# vLLM System Benchmarks

## Scope

End-to-end overhead measurement of steering in real vLLM inference. These produce the headline numbers for the article.

### Phase C migration status

`bench_latency.py`, `bench_throughput.py`, and `bench_memory.py` were modernized in Phase C:

- No `sys.path.insert` hack — they rely on the installed `steering_bench` package.
- Model dims come from `steering_bench.harness.models.get_model_config` (single source of truth; the per-script `MODEL_CONFIGS` tables were removed). Known models resolve offline to identical dims; unknown models resolve via HF `AutoConfig` or raise a clear `ModelConfigError` (the old silent gemma-dims fallback is gone).
- Each takes `--engine` (choices from the engine registry). Because these scripts exercise vLLM-fork-only behavior — `named_shared`/`inline_unique` modes, `register_steering_modules` collective RPC, `enable_prefix_caching`, per-request `max_steering_configs`, `num_gpu_blocks_override` — they are **guarded to `--engine vllm`** and exit with a clear error for any other engine (strategy (b): fork-specific path kept intact rather than deleting capability). Generation therefore still goes through the direct vLLM `LLM`/`SamplingParams` API, not the `SteeringEngine.generate` seam.
- Results are stamped with the first-class `engine` block via `write_result(engine=VllmSteeringEngine().identity())`.

The **engine-agnostic** slice of the latency measurement (single/batched steered latency at one layer/hook) is available through the harness: `python -m steering_bench run latency --engine <engine>`.

### Phase 3: engine-agnostic modes now first-class in the harness

The offline steering **modes** these scripts pioneered (`disabled`, `enabled_idle`, `inline_shared`, `inline_unique`, `named_shared`, `per_request_N`) are now expressed engine-agnostically over the seam (`src/steering_bench/harness/benchmarks/modes.py`) and driven by two harness benchmarks:

- `python -m steering_bench run latency --mode <mode> --engine <engine>` — per-iteration steered latency.
- `python -m steering_bench run throughput --mode <mode> --engine <engine>` — tokens/sec (input+output).

`named_shared` uses `engine.register_module` + `NamedModuleRef`, load-time knobs go through the typed `SteeringConfig` (`enable_steering`/`max_steering_configs`/`enable_prefix_caching`), and on engines lacking `named_modules` the `named_shared` mode degrades to inline-shared. `scripts/bench_latency.py`/`bench_throughput.py` are **retained** (not deleted in Phase 3) because they still carry fork-specific nuances the seam doesn't express yet — prefix-cache isolation (`--disable-prefix-cache` sweeps as a first-class axis), auto-promote steady-state semantics, per-mode `max_steering_configs` overrides, and the internal mode×batch matrix loop. Full script retirement / mechanical migration is Phase 7.

### In Scope
- Per-request latency: steering off/idle/per_request at various batch sizes
- Batch throughput: tokens/sec with varying distinct steering config counts
- GPU memory: delta cost of steering buffers at varying max_steering_configs

### Not In Scope
- Online serving via HTTP (complex subprocess management, deferred)
- Startup time measurement (not compelling for article)
- External library comparison (separate feature)

## Data/Control Flow

### bench_latency.py
```
For each mode in [disabled, enabled_idle, per_request_1, per_request_4]:
  Load vllm.LLM(enable_steering=...)
  For each batch_size:
    Create prompts + SamplingParams (with/without steering_vectors)
    Warmup: 5 generate() calls
    Measure: 20 generate() calls, wall-clock each
    Compute overhead_pct vs disabled baseline
    write_result("vllm.latency", ...)
  Unload model, gc.collect(), cuda.empty_cache()
```

Steering modes:
- `disabled`: enable_steering=False (baseline)
- `enabled_idle`: enable_steering=True, no vectors (zero-path overhead)
- `per_request_1`: one steering config via SamplingParams
- `per_request_4`: four distinct configs distributed across batch

### bench_throughput.py
```
For each distinct_configs in [0, 1, 4, 8]:
  Load vllm.LLM(enable_steering=configs>0)
  Create 64 prompts with round-robin steering configs
  Warmup: 3 generate() calls
  Measure: 5 generate() calls
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / elapsed_seconds
  write_result("vllm.throughput", ...)
```

### bench_memory.py
```
For each max_configs in [0, 4, 8, 16, 32]:
  torch.cuda.reset_peak_memory_stats()
  Load model with max_steering_configs=N
  Record torch.cuda.memory_allocated()
  Compute delta vs configs=0 baseline
  Compare against theoretical formula:
    per_layer = 3 * (max_configs + 3) * hidden_size * 4 bytes
    total = num_layers * per_layer + max_tokens * 8 bytes
  write_result("vllm.memory", ...)
```

### bench_steering_with_capture.py
```
For each steering_mode in [disabled, enabled_idle, per_request_1, per_request_4]:
  For each capture_mode in [cap_off, cap_on_idle, cap_on_active]:
    For each batch_size:
      In a fresh subprocess (run_in_subprocess):
        Load vllm.LLM(enable_steering=..., capture_consumers=...)
        Build SamplingParams with optional steering_vectors and optional
          per-request capture={"logging": {...}}
        Warmup: 3 generate() calls
        Measure: 10 generate() calls, wall-clock each
      Compute overhead_pct vs cap_off (per-steering-mode baseline)
      write_result("vllm.steering_with_capture", ...)
```

Capture modes:
- `cap_off`: no consumers, capture system inactive (cold path; the
  `maybe_capture_residual` gate is supposed to constant-fold out of the
  compiled graph)
- `cap_on_idle`: a logging consumer registered globally on (post_mlp,
  layer L) but no per-request `capture` field — manager installed but
  no per-request work
- `cap_on_active`: same logging consumer + per-request
  `capture={"logging": {...}}` so the manager actually gathers and
  dispatches a row per request

Each (steering_mode, batch_size) cell uses cap_off as its baseline; the
overhead_pct column reports cap_on_idle / cap_on_active relative to
cap_off at the same steering_mode and batch_size. This isolates the
"does enabling capture-consumers slow steering down?" question from the
unrelated steering-vs-disabled question already covered by
bench_latency.py.

Subprocess isolation is on by default: vLLM's in-process teardown leaks
weight memory, and a 4×3×N matrix rapidly OOMs without it. Pass
`--no-subprocess` to opt out for small sweeps.

## Files

| File | Purpose | Key CLI Args |
|------|---------|-------------|
| `scripts/bench_latency.py` | Latency overhead | `--model`, `--batch-sizes`, `--iters` |
| `scripts/bench_throughput.py` | Throughput impact | `--model`, `--num-prompts`, `--configs-sweep` |
| `scripts/bench_memory.py` | Memory cost | `--model`, `--configs-sweep` |
| `scripts/bench_steering_with_capture.py` | Steering latency × capture-on/off | `--model`, `--steering-modes`, `--capture-modes`, `--capture-layer`, `--no-subprocess` |

## Invariants

- bench_latency.py always runs `disabled` mode first to establish baseline
- bench_steering_with_capture.py uses `cap_off` as the baseline for each
  (steering_mode, batch_size) pair; if the user removes `cap_off` from
  `--capture-modes` the overhead column is omitted rather than computed
  against an arbitrary other capture mode
- All scripts handle OOM gracefully (catch, report, continue)
- Model is fully unloaded between configurations (gc.collect + cuda.empty_cache;
  bench_steering_with_capture additionally runs each cell in a fresh subprocess)
- Default model is google/gemma-3-4b-it (hidden_size=2560, num_layers=34)
- Latency measured with wall-clock (time.perf_counter) since LLM.generate() handles GPU sync
