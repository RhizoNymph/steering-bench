# vLLM System Benchmarks

## Scope

End-to-end overhead measurement of steering in real vLLM inference. These produce the headline numbers for the article.

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

## Files

| File | Purpose | Key CLI Args |
|------|---------|-------------|
| `scripts/bench_latency.py` | Latency overhead | `--model`, `--batch-sizes`, `--iters` |
| `scripts/bench_throughput.py` | Throughput impact | `--model`, `--num-prompts`, `--configs-sweep` |
| `scripts/bench_memory.py` | Memory cost | `--model`, `--configs-sweep` |

## Invariants

- bench_latency.py always runs `disabled` mode first to establish baseline
- All scripts handle OOM gracefully (catch, report, continue)
- Model is fully unloaded between configurations (gc.collect + cuda.empty_cache)
- Default model is google/gemma-3-4b-it (hidden_size=2560, num_layers=34)
- Latency measured with wall-clock (time.perf_counter) since LLM.generate() handles GPU sync
