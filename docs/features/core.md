# Core Infrastructure

## Scope

Shared utilities imported by all benchmark scripts.

### In Scope
- GPU-synchronized timing (CUDA events and CPU perf_counter)
- JSON result schema with environment metadata capture
- Reproducible steering vector generation
- Statistical aggregation (mean, median, percentiles, stddev)

### Not In Scope
- CLI framework (benchmarks are plain scripts with argparse)
- Model loading (each benchmark handles its own model lifecycle)
- Result analysis or plotting (future feature)

## Data/Control Flow

```
Benchmark Script
  ├─ timing.cuda_timer(warmup, iters, fn) 
  │    → runs warmup iters (unmeasured)
  │    → records CUDA event pairs for each measured iter
  │    → returns TimingStats(mean, median, p10..p99, samples)
  │
  ├─ timing.cpu_timer(warmup, iters, fn)
  │    → same but with time.perf_counter()
  │
  ├─ vectors.random_steering_vectors(hidden_size, num_layers, ...)
  │    → returns {hook_point: {layer_idx: [floats]}}
  │    → deterministic via seed parameter
  │
  └─ output.write_result(benchmark, params, results, output_dir)
       → captures environment (GPU, CUDA, torch, vllm versions)
       → writes timestamped JSON file
       → returns Path to written file
```

## Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `src/steering_bench/__init__.py` | Package marker | — |
| `src/steering_bench/timing.py` | GPU/CPU timing | `cuda_timer`, `cpu_timer`, `compute_stats`, `TimingStats`, `cuda_sync_timer` |
| `src/steering_bench/output.py` | Result output | `write_result`, `print_result_summary`, `capture_environment` |
| `src/steering_bench/vectors.py` | Vector generation | `random_steering_vectors`, `random_steering_vectors_diverse` |

## Invariants

- All timing uses explicit `torch.cuda.synchronize()` barriers for GPU measurements
- All random vectors are seeded for reproducibility (default seed=42)
- Every JSON result includes full environment metadata
- TimingStats always reports all percentiles regardless of sample count
- Vector scale defaults to 0.1 to avoid destabilizing model generation
