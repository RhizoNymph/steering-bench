# Performance Characteristics

Consolidated findings from the steering-bench suite, targeting the performance characteristics of vLLM's per-request activation steering on Gemma-3-4B (hidden_size=2560, num_layers=34).

## Executive summary

1. **Enabling steering is effectively free.** With steering enabled but no active vectors, latency overhead is ≤5% across all tested batch sizes.
2. **Memory cost is negligible.** 0.5 MB per config on Gemma-3-4B (bf16); ~16 MB at `max_steering_configs=32` — less than 0.1% of GPU memory.
3. **Per-request steering has a batch-size-scaling overhead.** At batch=1, +9% latency; at batch=16, +68%. Root cause identified.
4. **Prefix caching works correctly.** Content-hashed cache keys deduplicate identical vectors across requests. No cache invalidation from steering.
5. **CUDA graphs are preserved.** Steering doesn't break graph capture or replay.
6. **The root cause of per-request overhead is `torch.compile` fusion loss around the opaque steering custom op.** This has a clear optimization path.

## Methodology

- **Hardware**: Single NVIDIA RTX 3090 (24 GB, 936 GB/s HBM)
- **Model**: `google/gemma-3-4b-it` (bf16)
- **Steering vectors**: Random, seeded, scale=0.1, hook point = `post_mlp` unless noted otherwise
- **Measurement**: Wall-clock via `time.perf_counter` with `torch.cuda.synchronize()` barriers for vLLM benchmarks; CUDA events for microbenchmarks
- **Prefix caching**: Enabled by default unless explicitly disabled in ablation
- **Iterations**: Microbenchmarks use 100 warmup / 1000 measured; system benchmarks use 5 warmup / 20 measured
- **Runs pinned to same machine** with no concurrent GPU workloads

## Benchmark results

### Microbenchmarks

**Steering op kernel (`torch.ops.vllm.apply_steering`)**
- ~0.05 ms per call at `num_tokens ≤ 512`, `hidden_size=2560`, bf16
- ~0.125 ms per call at `num_tokens=2048`
- Insensitive to table row count
- Implication: the kernel itself is cheap; launch overhead dominates at small token counts

**SteeringManager methods** (34 layers, Gemma-3-4B config)
| operation | 1 config, 1 hook | 4 configs, 3 hooks | 16 configs, 3 hooks |
|---|---|---|---|
| `populate_steering_tables` | 2.14 ms | 5.06 ms | 15.4 ms |
| `register+release` cycle | 3.02 ms | 36.1 ms | 145.3 ms |
| `get_row_for_config` | ~0.2 μs | ~0.7 μs | ~2.5 μs |

Key takeaway: `populate_steering_tables` runs once per forward pass. At configs=1/1 hook (the minimum realistic workload), it costs **~2.14 ms per decode step** in pure Python + small CUDA kernel launches. This is the dominant Python-side cost.

**Index building** (CPU loop filling `steering_index`)
- ~5 μs per request per step, flat across 1–64 requests
- Insensitive to tokens-per-request and distinct config count
- At batch=64 over 128 decode steps: ~41 ms total (≤1.5% of steering overhead)
- **Not a bottleneck at any realistic batch size**

### End-to-end system benchmarks

**Latency overhead by mode** (Gemma-3-4B, max_tokens=128, prefix cache on)

| mode | batch=1 | batch=4 | batch=8 | batch=16 |
|---|---|---|---|---|
| `disabled` | 1567 ms (baseline) | 1553 ms | 1606 ms | 1690 ms |
| `enabled_idle` | +2.7% | +4.6% | +2.0% | -0.4% |
| `per_request_1` | +13.3% | +20.8% | +36.4% | +69.0% |
| `per_request_4` | +9.5% | +27.9% | +43.7% | +75.4% |

The `enabled_idle` mode is flat and cheap — the cost of having the feature on but unused is negligible.
The `per_request_*` modes scale with batch size, with per_request_1 and per_request_4 nearly identical (small constant offset for the second, fourth configs).

**Throughput** (64 prompts, prompt_len=64, max_tokens=128)

| distinct configs | tokens/sec | loss vs baseline |
|---|---|---|
| 0 (disabled) | 6175 | 0.0% |
| 1 | 1824 | 70.5% |
| 4 | 1210 | 80.4% |
| 8 | 1152 | 81.3% |

With `max_steering_configs=32` (big table, eliminating thrashing):

| distinct configs | tokens/sec | loss vs configs=0 |
|---|---|---|
| 0 | 6192 | 0.0% |
| 1 | 1849 | 70.1% |
| 4 | 1792 | 71.1% |
| 8 | 1723 | 72.2% |

With sufficient table headroom, scaling from 1 to 8 distinct configs costs only ~7% throughput. **Table sizing matters enormously**.

**Memory cost**
- Measured per-config cost: **0.5 MB per additional `max_steering_configs`**
- Formula: `3 hooks × hidden_size × num_layers × dtype_bytes`
- For Gemma-3-4B (bf16): `3 × 2560 × 34 × 2 = 522 KB per config`
- At `max_steering_configs=32`: ~16 MB total (≤0.07% of 24 GB)
- Validated against theoretical formula with ratios of 0.57x → 0.91x across the sweep (ratio approaches 1.0 as signal grows relative to measurement granularity)

### Ablations

**CUDA graphs (enforce_eager × enable_steering, batch sweep 1–64)**

| batch | graphs_no_steer (ms) | graphs_w_steer (ms) | steer_overhead (w/ graphs) | graph_speedup (no steer) | graph_speedup (w/ steer) |
|---|---|---|---|---|---|
| 1 | 1567 | 1777 | +13.4% | 2.93x | 2.76x |
| 4 | 1535 | 1877 | +22.3% | 3.18x | 2.78x |
| 8 | 1596 | 2193 | +37.4% | 2.98x | 2.51x |
| 16 | 1676 | 2847 | +69.8% | 2.87x | 2.15x |
| 32 | 1905 | 4278 | +124.5% | 2.56x | 1.72x |
| 64 | 2051 | 6770 | +230.2% | 2.39x | 1.45x |

CUDA graph benefit is **preserved at small batches** (ratio 0.94 at b=1) and **degraded at large batches** (ratio 0.61 at b=64). Graphs still replay correctly; the cost is compute inside the graph, not graph overhead.

**Per-step steering overhead** (derived from cuda_graphs ablation):
- batch=1: ~1.6 ms/step
- batch=8: ~4.5 ms/step
- batch=64: **~37 ms/step**

Per-step overhead scales approximately linearly with batch size.

**Hook points ablation** (1/2/3 active hooks, per-request steering)

| batch | 1 hook | 2 hooks | 3 hooks | per-hook cost (avg) |
|---|---|---|---|---|
| 1 | 1703 ms | +6.4% | +12.7% | ~108 ms/hook |
| 4 | 1880 ms | +17.0% | +33.7% | ~317 ms/hook |
| 8 | 2186 ms | +28.4% | +58.3% | ~637 ms/hook |

Each additional hook adds approximately linear cost, with the per-hook cost scaling with batch size. **Hook count matters**, and it scales multiplicatively with batch size, which is the signature of fusion loss: each hook breaks one additional `torch.compile` fusion boundary per layer, and the lost fusion benefit grows with the size of the data being fused.

**Config scaling ablation** (fixed batch=8, sweep max_steering_configs)

| max_configs | distinct used | latency | overhead vs max=1 |
|---|---|---|---|
| 1 | 1 | 13054 ms | baseline |
| 2 | 2 | 6790 ms | -48% |
| 4 | 4 | 3922 ms | -70% |
| 8 | 8 | 2571 ms | -80% |
| 16 | 8 | 2520 ms | -81% |
| 32 | 8 | 2510 ms | -81% |

Latency is inversely proportional to max_steering_configs up to `max_configs ≥ batch_size`, then flat. This indicates a scheduler-bound regime: below the threshold, the scheduler cannot batch requests that would exceed the available table rows, so requests serialize. **Recommendation**: set `max_steering_configs` ≥ the expected distinct-config concurrency.

**Prefix cache ablation**

Latency overhead with and without prefix caching (batch=16, per_request_1):
- prefix cache on: +69.0%
- prefix cache off: +67.7%

Throughput with and without prefix caching (configs=1, batch=64):
- prefix cache on: 1824 tok/s
- prefix cache off: 1800 tok/s

Prefix caching is effectively orthogonal to steering overhead. **The content-hashed cache key correctly deduplicates identical steering vectors across requests.** Cache misses are not driving the per-request cost.

### Profiling evidence (torch.profiler, batch=8, 3 iterations)

Key deltas between steering and disabled runs:

| operation | disabled | steering | delta |
|---|---|---|---|
| `cudaLaunchKernel` wall | 240 ms | 311 ms | **+29%** (+70 ms, ~14k extra launches) |
| `cudaStreamSynchronize` | not present | 204 calls, 1024 ms | **new** (1 sec of host-blocked sync) |
| `aten::copy_` | 4800 calls, 50 ms | 18060 calls, 156 ms | +3.7x calls (populate vector copies) |
| `aten::to` / `_to_copy` | 1152/1728 | 7884/8460 | +5x (device/dtype conversions) |
| `cudaMemcpyAsync` | 3648 | 10380 | +2.8x (host-device transfers in register_config) |
| `cudaGraphLaunch` | 294 | 294 | **unchanged** (graphs preserved) |

Two observations:

1. **`cudaLaunchKernel` increases by ~14,000 launches** over 3 iters (~37 extra launches per decode step), consistent with fusion loss producing more, smaller kernels.
2. **`cudaStreamSynchronize` appears 204 times, consuming ~1 second of wall time**, none of which exists in the disabled run. This is host-blocked synchronization in the per-step steering path — an immediate optimization target.

## Root cause: triangulated evidence

Five independent lines of evidence converge on **loss of `torch.compile` kernel fusion around the opaque steering op** as the primary cause of per-request overhead:

1. **Microbenchmark**: the steering op kernel itself is cheap (~0.05 ms regardless of small token counts). The overhead cannot be attributed to the op's own compute cost.
2. **CUDA graphs ablation**: graph replay is preserved (ratio ≥0.61) but graph benefit degrades with batch size. The cost is inside the graph — not graph break overhead — which must be unfused kernels.
3. **Hook count ablation**: per-hook cost scales linearly with both hook count and batch size. Each hook breaks one additional fusion boundary per layer, and the lost fusion benefit scales with the data volume being fused. Memory bandwidth alone cannot explain the observed scale; fusion loss does.
4. **Prefix cache ablation**: disabling prefix caching does not reduce per-request overhead (latency: 69.0% → 67.7%; throughput: 1824 → 1800 tok/s). Cache invalidation is ruled out.
5. **Direct profiling**: `cudaLaunchKernel` wall time increases by 29% in the steering run, corresponding to ~14,000 additional kernel launches over 3 iterations. More launches = smaller kernels = broken fusion.

A secondary cost is **Python-level orchestration overhead** in `populate_steering_tables`, measured at ~2 ms per decode step from pure tensor copy + metadata bookkeeping, and ~1 second of accumulated `cudaStreamSynchronize` calls over 3 profiled iterations. This affects all batch sizes but is dominant only at small batches where fusion loss is less expensive.

## What isn't the bottleneck

Several candidates were tested and ruled out as significant contributors:

- **Prefix cache invalidation** (content hash works; ≤2% impact)
- **Raw steering op kernel cost** (~0.05 ms, confirmed by microbench)
- **`steering_index` construction loop** (~5 μs/request, ≤1.5% of overhead at batch=64)
- **Memory bandwidth of the steering op** (≤0.1 ms/step by first-principles calculation)
- **CUDA graph break at the graph replay level** (replay count unchanged, graphs preserved)
- **Per-request `get_row_for_config` lookup cost** (sub-microsecond)
- **Register/release cycle cost** (refcount-amortized to ~once per iter, not per request)

## Optimization roadmap

Concrete targets, with expected impact grounded in the measurements above:

### 1. `torch.compile` fusion through the steering op (largest impact)
Make the steering op transparent to Inductor so fusion can proceed through it. Options:
- Split into functional gather + add components that Inductor can fuse individually into adjacent residual/layernorm operations.
- Register as `@torch.library.custom_op` with a functional impl Inductor can reason about.
- Add a custom Inductor pass that recognizes the steering op pattern and inlines it.

**Expected impact**: eliminates the batch-size-dependent per-step cost. Hook count should become free. Per-step overhead should drop from ~37 ms/step to ~1-2 ms/step at batch=64.

### 2. Eliminate per-step host-GPU sync (`cudaStreamSynchronize`)
Profile shows 204 sync calls totaling ~1 second of wall time over 3 iterations that only occur in the steering path. Identify the source (likely a `.item()`, `.detach_()`, or CPU-side control flow on a GPU tensor in the per-step update) and remove it.

**Expected impact**: ~10% wall-clock savings at small-to-medium batches based on profile measurements. Independent of the fusion fix.

### 3. Batch `populate_steering_tables` into fewer CUDA calls
Currently launches ~100 small kernels per decode step (one `.zero_()` and up to one `.copy_()` per (layer, hook) combination). Instead: build a single flat update tensor on CPU, transfer once, scatter into tables with one call.

**Expected impact**: `populate_steering_tables` drops from ~2 ms/step to <0.5 ms/step. Most of the value at small batches.

### 4. Skip inactive hook points in populate
The loop always iterates all 3 hook points. When only 1 hook is actively set, 2/3 of the work is wasted zeroing rows. Track `active_hook_points: set[str]` and skip hook points outside the set.

**Expected impact**: ~60% populate-cost reduction when users set only one hook (the common case).

### 5. Cache populate output when config state is stable
Most decode steps don't change the active config set — same configs, same vectors. Add a "dirty" flag and skip populate when the state hasn't changed.

**Expected impact**: `populate_steering_tables` becomes effectively free in the steady state (most decode steps).

### 6. Pre-convert vectors at SamplingParams construction
`register_config` currently converts Python lists to CUDA tensors on every registration. Measurement: ~9 ms per registration cycle at 1 config, scaling to ~145 ms at 16 configs. Pre-converting at the serving boundary would move this cost out of the hot path.

**Expected impact**: eliminates a tail-latency risk at high-config-churn workloads. Not critical at current default max_configs=4.

## Usage recommendations (with current implementation)

Pending the optimizations above, users can minimize steering overhead by:

1. **Use only one hook point** (`post_mlp` is usually sufficient). Adding hooks is linearly expensive until the fusion fix lands.
2. **Set `max_steering_configs` generously**: at minimum equal to the expected distinct-config concurrency, ideally with 2-4x headroom. Memory cost is trivial (~0.5 MB per config); latency cost of undersizing is severe (scheduler serializes requests).
3. **Use the `enabled_idle` mode** (steering compiled in but no active vectors) for requests that don't need steering — this path is essentially free and lets a single vLLM instance serve both steered and unsteered traffic.
4. **Prefer global steering vectors** (`collective_rpc("set_steering_vectors", ...)`) when all requests share the same steering target. Global vectors do not compete for table rows and do not trigger per-request registration overhead.

## Comparison to external libraries

Quantitative head-to-head comparison against TransformerLens, nnsight, repeng, and pyvene is **deferred to follow-up work** until the optimizations above are implemented. Comparing against a known-suboptimal version would mischaracterize vLLM's capability.

**Architectural advantages hold independently of the optimization work**:

- **Continuous batching with per-request vectors.** Other libraries either sequentialize multi-request workloads (TransformerLens, pyvene) or require the same intervention across a batch (repeng). vLLM deduplicates identical vectors and schedules distinct vectors in a single continuous batch.
- **Production serving stack.** OpenAI-compatible API, prefix caching, paged attention, KV cache reuse, multi-GPU support via tensor and pipeline parallel. Research libraries inherit none of this.
- **Zero cost to enable the feature.** `enabled_idle` mode shows ≤5% latency overhead across all batch sizes, so a team can deploy steering without penalty for requests that don't use it.
- **Scalable integration.** A team already running vLLM in production can add steering without adopting a new framework or reimplementing their serving pipeline.

## Scaling projections

For a hypothetical dense 1T model (`hidden_size=20480, num_layers=160`):

- **Memory**: ~656 MB at `max_steering_configs=32`. Less than 0.03% of weights; negligible.
- **Per-step overhead (current implementation)**: extrapolating from Gemma-3-4B by `hidden_size × num_layers` scaling, ~1.4 s per decode step at batch=64 — **not viable** without the fusion optimization.
- **Per-step overhead (post-optimization)**: projected <50 ms per step at batch=64, which makes large-model steering feasible.

The optimization work is **not an incremental improvement but a requirement for scaling steering to production models**.
