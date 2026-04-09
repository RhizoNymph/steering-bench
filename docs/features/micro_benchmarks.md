# Micro Benchmarks

## Scope

Time the raw steering primitives without running full model inference. Establishes the floor cost of steering.

### In Scope
- `torch.ops.vllm.apply_steering` kernel latency across parameter sweeps
- `SteeringManager` method timing (register, release, populate, get_row)
- `steering_index` construction loop timing

### Not In Scope
- Full model inference (that's vllm_benchmarks)
- Interaction with CUDA graphs or torch.compile (that's ablation_benchmarks)

## Data/Control Flow

### bench_steering_op.py
```
Create GPU tensors (hidden_states, steering_table, steering_index)
  → Sweep: hidden_size x num_tokens x table_rows x dtype
    → cuda_timer(100 warmup, 1000 iters, apply_steering, ...)
    → write_result("micro.steering_op", ...)
```

Falls back to reference implementation if vLLM custom op is unavailable.

### bench_steering_manager.py
```
Create mock decoder layers with register_steering_buffers()
Create SteeringManager
  → Sweep: num_layers x num_configs x hook_points
    → cpu_timer: populate_steering_tables (the per-step critical path)
    → cpu_timer: get_row_for_config (the per-request lookup)
    → cpu_timer: register/release cycle
    → write_result("micro.steering_manager", ...)
```

### bench_index_building.py
```
Create steering_index tensor on GPU
  → Sweep: num_requests x tokens_per_request x config_type
    → cpu_timer: build_index_uniform (write row indices for each request)
    → cpu_timer: build_index_mixed_phase (prefill + decode mix)
    → write_result("micro.index_building", ...)
```

## Files

| File | Purpose |
|------|---------|
| `scripts/bench_steering_op.py` | Kernel latency sweep |
| `scripts/bench_steering_manager.py` | Manager method overhead |
| `scripts/bench_index_building.py` | Index construction loop |

## Invariants

- Steering op benchmark uses 100 warmup / 1000 measured iters for statistical power
- Manager and index benchmarks use 50 warmup / 500 measured iters
- The `--subset` flag on bench_steering_op.py runs a reduced sweep (~6 configs vs ~120)
- bench_steering_op.py works without vLLM installed (uses reference implementation)
- bench_steering_manager.py requires vLLM installed (imports SteeringManager)
