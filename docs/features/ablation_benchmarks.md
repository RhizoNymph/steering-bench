# Ablation Benchmarks

## Scope

Isolate how vLLM optimizations interact with steering.

### In Scope
- CUDA graphs: 2x2 matrix (enforce_eager x enable_steering)

### Not In Scope (deferred)
- torch.compile levels
- Prefix caching interaction
- Config count scaling
- Hook point count scaling
- Cross-model-family comparison

## Data/Control Flow

### bench_cuda_graphs.py
```
For each (enforce_eager, enable_steering) in 2x2 matrix:
  Load vllm.LLM(enforce_eager=..., enable_steering=...)
  For each batch_size:
    Warmup: 5 generate() calls
    Measure: 20 generate() calls
    write_result("ablation.cuda_graphs", ...)
  Unload model

Derive:
  graph_speedup_no_steer  = eager_no_steer / graphs_no_steer
  graph_speedup_w_steer   = eager_w_steer / graphs_w_steer
  steer_overhead_w_graphs = (graphs_w_steer - graphs_no_steer) / graphs_no_steer
  steer_overhead_eager    = (eager_w_steer - eager_no_steer) / eager_no_steer
  interaction             = graph_speedup_w_steer / graph_speedup_no_steer
    → >0.9 means CUDA graph benefit is preserved with steering
```

## Files

| File | Purpose |
|------|---------|
| `scripts/bench_cuda_graphs.py` | CUDA graphs x steering 2x2 ablation |

## Invariants

- Runs 4 model loads (one per matrix cell) per batch size
- Steering configs use per-request vectors (not global) for realistic measurement
- Reports interaction effect: whether steering degrades CUDA graph benefit
- "preserved" verdict if interaction ratio > 0.9
