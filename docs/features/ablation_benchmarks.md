# Ablation Benchmarks

## Scope

Isolate how vLLM optimizations interact with steering.

### In Scope
- CUDA graphs: 2x2 matrix (enforce_eager x enable_steering)
- Config scaling: 1x6 sweep (max_steering_configs 1-32)
- Hook points: 1x3 sweep (1/2/3 active hooks)

### Not In Scope (deferred)
- torch.compile levels
- Prefix caching interaction
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

### bench_config_scaling.py
```
For each max_steering_configs in [1, 2, 4, 8, 16, 32]:
  Load vllm.LLM(enable_steering=True, max_steering_configs=N)
  actual_distinct = min(N, batch_size)
  Create actual_distinct diverse steering vectors, distribute round-robin
  Warmup + measure latency
  Record allocated_mb
  write_result("ablation.config_scaling", ...)

Derive: overhead_pct vs configs=1 baseline
```

### bench_hook_points.py
```
For each hook_config in [1_hook, 2_hooks, 3_hooks]:
  Load vllm.LLM(enable_steering=True)
  Generate vectors only for active hooks (others stay zero in table)
  For each batch_size:
    Warmup + measure latency
    write_result("ablation.hook_points", ...)

Derive: overhead vs 1-hook baseline
Verdict: "negligible" if 3-hook vs 1-hook difference < 5%
```

## Files

| File | Purpose |
|------|---------|
| `scripts/bench_cuda_graphs.py` | CUDA graphs x steering 2x2 ablation |
| `scripts/bench_config_scaling.py` | max_steering_configs scaling (1x6 sweep) |
| `scripts/bench_hook_points.py` | Active hook point count (1x3 sweep) |

## Invariants

- CUDA graphs: 4 model loads per batch size; "preserved" verdict if interaction ratio > 0.9
- Config scaling: fixed batch_size=8 to isolate table-size variable; re-loads model per config value
- Hook points: re-loads model per hook config; omitted hooks stay zero via SteeringManager behavior
- All ablations use per-request vectors (not global) for realistic measurement
