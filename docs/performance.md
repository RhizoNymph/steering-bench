# Performance Characteristics

Consolidated findings from the steering-bench suite, targeting the performance characteristics of vLLM's per-request activation steering on Gemma-3-4B (hidden_size=2560, num_layers=34).

## Executive summary

1. **Enabling steering is effectively free.** With steering enabled but no active vectors, latency and throughput overhead is within measurement noise (-1.9% to +2.1%) across all tested batch sizes. Indistinguishable from the disabled baseline.
2. **Memory cost is negligible.** 0.5 MB per config on Gemma-3-4B (bf16); ~16 MB at `max_steering_configs=32` — less than 0.1% of GPU memory.
3. **Mixed-batch cost scales proportionally with active requests, not transitively.** Non-steered requests in a batch containing some steered requests pay only a small proportional share of the steering overhead. A workload with 5% steering traffic pays ~a few percent instance-wide overhead, not the ~74% all-active cost. Shared multi-tenant deployment is viable.
4. **Per-step overhead fits a linear model: `cost ≈ 0.5 ms fixed + 0.58 ms × num_active`** (Gemma-3-4B, per decode step). The dominant term scales with the number of active steered requests, not with batch size. At batch=32 fully steered this amounts to +124% total latency.
5. **Prefix caching works correctly.** Content-hashed cache keys deduplicate identical vectors across requests. No cache invalidation from steering.
6. **CUDA graphs are preserved.** Steering does not break graph capture or replay; graph effectiveness degrades modestly at larger batches.
7. **Two optimization targets, not one.** Evidence supports (a) `torch.compile` fusion loss around the opaque custom op as a smaller fixed-per-step cost, and (b) a larger per-active-request cost whose exact mechanism has not been pinned down. Both are addressable independently.

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
| operation | 1 config, 1 hook | 1 config, 3 hooks | 4 configs, 3 hooks | 16 configs, 3 hooks |
|---|---|---|---|---|
| `populate_steering_tables` | 2.14 ms | — | 5.06 ms | 15.4 ms |
| `register+release` cycle | 3.02 ms | ~9 ms | 36.1 ms | 145.3 ms |
| `get_row_for_config` | ~0.2 μs | — | ~0.7 μs | ~2.5 μs |

Key takeaway: `populate_steering_tables` runs once per forward pass. At configs=1/1 hook (the minimum realistic workload), it costs **~2.14 ms per decode step** in pure Python + small CUDA kernel launches. This is the dominant Python-side cost.

**Index building** (CPU loop filling `steering_index`)
- ~5 μs per request per step, flat across 1–64 requests
- Insensitive to tokens-per-request and distinct config count
- At batch=64 over 128 decode steps: ~41 ms total (≤1.5% of steering overhead)
- **Not a bottleneck at any realistic batch size**

### End-to-end system benchmarks

**Throughput matrix: mode × batch_size** (Gemma-3-4B, max_tokens=128, prefix cache default)

The primary end-to-end measurement. Five steering modes swept across five batch sizes with one model load per (`enable_steering`) setting, so within-phase results are directly comparable.

Latency (ms):

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `disabled` | 1565 | 1524 | 1560 | 1635 | 1911 |
| `enabled_idle` | 1577 | 1555 | 1563 | 1633 | 1875 |
| `mixed_25` | — | 1726 | 1777 | 1979 | 2514 |
| `mixed_50` | — | 1737 | 1911 | 2255 | 3080 |
| `mixed_75` | — | 1804 | 2047 | 2556 | 3663 |
| `all_steered` | 1716 | 1871 | 2183 | 2839 | 4273 |

Throughput (tokens/sec):

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `disabled` | 123 | 504 | 985 | 1878 | 3219 |
| `enabled_idle` | 122 | 494 | 983 | 1881 | 3277 |
| `mixed_25` | — | 445 | 864 | 1552 | 2444 |
| `mixed_50` | — | 442 | 804 | 1362 | 1995 |
| `mixed_75` | — | 426 | 750 | 1202 | 1677 |
| `all_steered` | 112 | 411 | 704 | 1082 | 1438 |

Latency overhead (%) vs `disabled` at same batch size:

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `enabled_idle` | +0.7% | +2.1% | +0.2% | -0.2% | -1.9% |
| `mixed_25` | — | +13.3% | +13.9% | +21.0% | +31.6% |
| `mixed_50` | — | +14.0% | +22.5% | +37.9% | +61.2% |
| `mixed_75` | — | +18.4% | +31.2% | +56.3% | +91.7% |
| `all_steered` | +9.7% | +22.8% | +39.9% | +73.6% | +123.6% |

Throughput loss (%) vs `disabled` at same batch size:

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `enabled_idle` | -0.7% | -2.0% | -0.2% | +0.2% | +1.8% |
| `mixed_25` | — | -11.7% | -12.2% | -17.4% | -24.1% |
| `mixed_50` | — | -12.3% | -18.4% | -27.5% | -38.0% |
| `mixed_75` | — | -15.5% | -23.8% | -36.0% | -47.9% |
| `all_steered` | -8.8% | -18.6% | -28.5% | -42.4% | -55.3% |

Three key observations from this table:

1. **`enabled_idle` is statistically indistinguishable from `disabled`.** Overheads range from -1.9% to +2.1% with roughly equal sign distribution. Since `enabled_idle` cannot mechanically be faster than `disabled`, the negative values confirm the signal is zero and the variance is measurement noise. Enabling the steering feature without active vectors has no measurable cost.

2. **`all_steered` per-step overhead roughly doubles with each batch size doubling** (assuming ~128 decode steps: b=1 → 1.2 ms/step, b=4 → 2.7, b=8 → 4.9, b=16 → 9.4, b=32 → 18.5). In `all_steered` mode, `num_active = batch_size`, so this pattern is mathematically equivalent to either "cost ∝ batch_size" (fusion loss) or "cost ∝ num_active" (per-active scaling). These two hypotheses cannot be distinguished from the `all_steered` data alone. The mixed-batch data (below) distinguishes them and supports the second interpretation.

3. **At batch=32 fully steered, steering more than doubles total latency** (+123.6%). Per-request deployment without the optimizations described below is not viable at production batch sizes.

4. **Per-request latency overhead converges with batch size.** Computing `total_overhead / num_active` for the `all_steered` rows: b=1 → 151 ms/request, b=4 → 87, b=8 → 78, b=16 → 75, b=32 → 74. This asymptotically approaches ~74 ms per active request as the batch grows, consistent with the per-step linear model amortizing a small fixed per-step cost across more active requests at larger batches.

**Distinct-configs throughput** (64 prompts, prompt_len=64, max_tokens=128 — legacy measurement, superseded by the throughput matrix above for mode × batch analysis, but still the primary source for the `distinct_configs` dimension).

With `max_steering_configs=4` (default):

| distinct configs | tokens/sec | loss vs baseline |
|---|---|---|
| 0 (disabled) | 6175 | 0.0% |
| 1 | 1824 | 70.5% |
| 4 | 1210 | 80.4% |
| 8 | 1152 | 81.3% |

With `max_steering_configs=32` (ample table headroom):

| distinct configs | tokens/sec | loss vs configs=0 |
|---|---|---|
| 0 | 6192 | 0.0% |
| 1 | 1849 | 70.1% |
| 4 | 1792 | 71.1% |
| 8 | 1723 | 72.2% |

At batch=64 with sufficient table headroom, scaling from 1 to 8 distinct configs costs only ~7% extra throughput. The gap between `max_steering_configs=4` and `max_steering_configs=32` at `distinct=4` (1210 → 1792 tok/s, +48%) confirms **table sizing matters significantly** when the workload has more distinct configs than the table has room for; at `distinct=1` the bigger table provides no benefit, as expected.

**Memory cost**
- Measured per-config cost: **0.5 MB per additional `max_steering_configs`**
- Formula: `3 hooks × hidden_size × num_layers × dtype_bytes`
- For Gemma-3-4B (bf16): `3 × 2560 × 34 × 2 = 522 KB per config`
- At `max_steering_configs=32`: ~16 MB total (≤0.07% of 24 GB)
- Validated against theoretical formula with ratios of 0.57x → 0.91x across the sweep (ratio approaches 1.0 as signal grows relative to measurement granularity)

**Mixed-batch proportional scaling** (derived from the throughput matrix above)

A batch contains a mix of steered and non-steered requests. Theoretical expectation from continuous batching would be transitive cost: any active request forces the entire batch to pay the full per-request cost, because the forward pass is shared across all tokens. The measurements contradict this expectation.

Per-active-request marginal latency cost, computed from the throughput matrix as `(latency_with_n_active - latency_disabled) / n`:

| batch size | all_steered cost/active |
|---|---|
| 4 | 86.8 ms |
| 8 | 77.9 ms |
| 16 | 75.3 ms |
| 32 | 73.8 ms |

Cost per additional active request **converges to ~74 ms** at larger batch sizes and stays stable whether the active fraction is 25%, 50%, 75%, or 100%. Checked directly by computing successive deltas at batch=32:

- 0 → 8 active: +603 ms ÷ 8 = **75.4 ms/active**
- 8 → 16 active: +567 ms ÷ 8 = **70.8 ms/active**
- 16 → 24 active: +583 ms ÷ 8 = **72.9 ms/active**
- 24 → 32 active: +610 ms ÷ 8 = **76.3 ms/active**

**Per-step linear model.** Fitting `per_step_cost = fixed + C × num_active` across all `(batch, num_active)` cells from the throughput matrix (excluding `batch=1`, which is noise-dominated) yields:

`per_step_cost ≈ 0.5 ms fixed + 0.58 ms × num_active`

At 128 decode steps per iter, this translates to **~74 ms per active request per iter**, matching the per-active deltas above.

Per-batch-size fit quality check:

| batch | fit | per-active slope | fixed term |
|---|---|---|---|
| 32 | `0.13 + 0.57·n` | 0.57 | 0.13 ms |
| 16 | `0.45 + 0.56·n` | 0.56 | 0.45 ms |
| 8 | `0.64 + 0.53·n` | 0.53 | 0.64 ms |
| 4 | `1.20 + 0.38·n` | 0.38 | 1.20 ms |

The linear model is **clean at batch ≥ 16** (slope stable around 0.56–0.57) and **less clean at batch ≤ 8** (slope drops, intercept grows). At small batches measurement noise dominates and the fit is approximate. A small batch-proportional cost component cannot be completely ruled out from these fits — it would manifest as a slope that drifts with batch size — but the effect is much smaller than the per-active term.

**Crucially, there is no large batch-proportional term.** If fusion loss were the dominant cost, the per-active slope should *not* stay flat across batch sizes — fusion loss would manifest as cost scaling with total batch_size even at constant num_active, producing a clear batch-proportional contribution on top of the per-active term. The fits do not show this.

**Deployment implication:** A vLLM instance with `enable_steering=True` serving mixed traffic pays cost proportional to actual steering usage. For a workload where 5% of requests use steering, the effective instance-wide latency overhead is a few percent, not the ~74% all-active cost. Shared multi-tenant inference with steering is viable without penalizing non-steering users.

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
| `aten::detach_` | not present | 204 calls, 15 ms | **new** (same count as the syncs) |
| `aten::copy_` | 4800 calls, 50 ms | 18060 calls, 156 ms | +3.7x calls (populate vector copies) |
| `aten::to` / `_to_copy` | 1152/1728 | 7884/8460 | +5x (device/dtype conversions) |
| `cudaMemcpyAsync` | 3648 | 10380 | +2.8x (host-device transfers in register_config) |
| `cudaGraphLaunch` | 294 | 294 | **unchanged** (graphs preserved) |

Three observations:

1. **`cudaLaunchKernel` increases by ~14,000 launches** over 3 iters (~37 extra launches per decode step), consistent with some degree of fusion loss producing more, smaller kernels.
2. **`cudaStreamSynchronize` appears 204 times, consuming ~1 second of wall time**, none of which exists in the disabled run. One sync per decode step on average.
3. **`aten::detach_` appears exactly 204 times** — the same count as the syncs, suggesting both originate from the same code path. This pattern does not correspond to any explicit `detach_()` or synchronization call in the steering code; it likely reflects PyTorch's internal bookkeeping at a boundary specific to the steering execution path (candidate: implicit synchronization between eager `populate_steering_tables` writes and the subsequent CUDA graph replay that reads those buffers).

**Static code search.** A direct search of the steering code (`vllm/v1/worker/steering_manager.py`, `vllm/model_executor/layers/steering.py`, and the `_update_steering_buffers` method in `vllm/v1/worker/gpu_model_runner.py`) found **no explicit synchronization calls**: no `.item()`, no `.cpu()`, no `.tolist()`, no `.numpy()`, no `torch.cuda.synchronize()`, and no comparisons on GPU scalars in the hot path. All tensor operations (`.zero_()`, `.copy_()`, `.clone()`, `+`, `.to()`, `fill_`-style assignments) are async. The 204 syncs are therefore coming from PyTorch internals or implicit boundaries, not from a line of steering code we could simply delete.

## Root cause: two mechanisms

Earlier drafts attributed all per-request overhead to `torch.compile` fusion loss around the opaque steering op. The more complete analysis supported by the throughput matrix shows **two separable cost sources**, with different scaling and different root causes:

### Mechanism A: fixed per-step cost (~0.5 ms/step), candidate = fusion loss

This term is small, roughly independent of `num_active`, and present whenever steering is active.

Evidence **consistent with** fusion loss:
1. **CUDA graph ablation** shows graph benefit is degraded with steering (speedup ratio 2.93x → 2.76x at b=1, 2.39x → 1.45x at b=64). Graph replay is preserved; the cost is *inside* the graph, consistent with unfused kernels.
2. **Profiler** captures +14k extra `cudaLaunchKernel` calls in the steering run, corresponding to ~37 extra kernels per decode step — roughly what you'd expect from 3 hooks × ~12 unfused kernels each.
3. **Microbench** confirms the steering op kernel itself is cheap (~0.05 ms/call). Some other source must account for the fixed cost; unfused-neighbor kernels are one plausible source.

Evidence **against** fusion loss being the dominant cost:
1. The mixed-batch data shows **no significant batch-proportional term** in the per-step cost model. Pure fusion loss would produce unfused kernels that process the whole residual tensor (all batch_size tokens per step), producing a cost that scales with total batch, not with active count.
2. The hook count ablation's linear per-hook scaling is *consistent* with fusion loss but also consistent with per-hook kernel invocations that scale with batch size because of the per-active cost at all_steered mode (which is what that ablation tested).

**Status**: fusion loss likely contributes to the fixed ~0.5 ms/step term but is **not** the dominant cost. Direct kernel-level confirmation would come from an `nsys` trace comparing disabled vs steering at fixed batch size and comparing kernel counts / durations within the forward pass (see "Follow-up benchmarks" below).

### Mechanism B: per-active-request cost (~0.58 ms/step per active), mechanism unknown

This is the dominant cost at production batch sizes and is **not explained by pure fusion loss**. Fusion loss should scale with total batch size (because unfused kernels process the whole residual tensor). Instead, we observe cost scaling with `num_active` — the number of steered requests — largely independent of total batch size.

**Static code search rules out the obvious candidates.** A direct grep of `vllm/v1/worker/steering_manager.py`, `vllm/model_executor/layers/steering.py`, and the `_update_steering_buffers` method in `vllm/v1/worker/gpu_model_runner.py` found **no explicit synchronization calls** in the hot path: no `.item()`, no `.cpu()`, no `.tolist()`, no `.numpy()`, no `torch.cuda.synchronize()`, and no comparisons on GPU scalars. All tensor operations (`.zero_()`, `.copy_()`, `.clone()`, `+`, `.to()`, slice-fill assignments) are async. The per-request loop in `_update_steering_buffers` iterates over `num_reqs` (not `num_active`), so its Python cost is batch-proportional, not active-proportional — which means the per-active scaling is **not** coming from that loop either. The `.item()` calls found in the worker path are in `get_steering_status()`, a status-reporting API not called in the hot path.

**Candidate hypotheses** (none confirmed):

1. **Implicit synchronization between populate and graph replay.** `populate_steering_tables` launches ~100 eager kernels before each forward pass. The subsequent CUDA graph replay reads the same buffers. PyTorch / vLLM may be inserting implicit stream synchronization to enforce ordering. Profile shows 204 `cudaStreamSynchronize` and 204 `aten::detach_` events with matching counts (one pair per decode step), consistent with a per-step boundary condition. **Does not cleanly explain the proportional-to-active scaling**, since a per-step sync would be batch-size-proportional (or constant per-step), not active-count-proportional.
2. **Per-active work elsewhere in the engine.** Something in the sampler, detokenizer, or per-request state management may do per-active-config work. The V1 engine has not been audited outside the steering-specific files.
3. **GPU memory access patterns in the indexed gather** depend on index distribution. All tokens pointing to row 0 is maximally cache-friendly; mixed indices are less so. Speculative and does not obviously quantitatively match 0.58 ms/step per active.
4. **Something in continuous batching's treatment of steering-active requests** that sub-batches them differently from unsteered requests. Would require reading the V1 scheduler to confirm or refute.

**How to get ground truth.** The single experiment that would definitively identify the mechanism is an `nsys` timeline with NVTX annotations around `populate_steering_tables`, the per-request index build loop, and the model forward pass, run at two points: `(batch=16, num_active=4)` and `(batch=16, num_active=16)`. Whichever NVTX section's duration scales with `num_active` is the source. Estimated diagnostic time: ~30 min including adding the annotations and running nsys. Pinning this down is the prerequisite for targeting optimization #1 in the roadmap below.

### Other rule-outs

From direct measurement:

- **Prefix cache invalidation** — ruled out. Disabling prefix caching does not reduce per-request overhead (latency: 69.0% → 67.7%; throughput: 1824 → 1800 tok/s). The content-hashed cache key works correctly.
- **Raw steering op kernel cost** — ruled out at ~0.05 ms/call.
- **`steering_index` construction loop** — ruled out at ~5 μs/request (≤1.5% of overhead at batch=64).
- **`populate_steering_tables` Python overhead alone** — measured at ~2 ms/step in isolation. Can account for the fixed per-step component but **cannot** account for per-active-request scaling (the populate loop iterates `active_configs`, not `active_requests`).
- **`get_row_for_config` lookup cost** — sub-microsecond; ruled out.
- **Register/release cycle cost** — refcount-amortized to ~once per iter, not per request. Not in the hot path.
- **`cudaLaunchKernel` launch overhead alone** — only 70 ms total delta over 3 iters; at most a small fraction of the observed per-active cost.

## Optimization roadmap

Concrete targets, ordered by expected impact. The top two correspond to the two root-cause mechanisms; the remainder are tractable improvements grounded in the measurements above.

### 1. Identify the per-active-request cost source (highest impact)

**Status**: mechanism unknown. The per-step overhead fit is `0.58 ms × num_active`. At batch=32 this is 18.5 ms/step or ~2.4 s per iter — the dominant cost at production batch sizes.

**Diagnostic approach**: add NVTX annotations around suspected sections and profile with `nsys`:

```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("populate_steering_tables"); ...; nvtx.range_pop()
nvtx.range_push("build_steering_index"); ...; nvtx.range_pop()
nvtx.range_push("model_forward"); ...; nvtx.range_pop()
```

The NVTX timeline in nsys will show exactly which section scales with `num_active` and where the time goes. Once identified:

- If it's an implicit sync between eager populate and graph replay → restructure populate to use a CUDA stream that the graph can wait on, or convert populate itself to a captured operation.
- If it's per-active work elsewhere in the engine → refactor that path.
- If it's GPU memory access pattern dependent → pre-sort or pack the steering index to be cache-friendly.

**Expected impact**: if the proportional term is fully eliminated, per-step overhead drops from ~18.5 ms at b=32 all-steered to ~0.5 ms (the fixed cost only). That's ~97% reduction at production batch sizes.

### 2. `torch.compile` fusion through the steering op

Responsible for the ~0.5 ms/step fixed cost under the two-mechanism model. Make the steering op transparent to Inductor so fusion can proceed through it:

- Split into functional gather + residual-add components that Inductor can fuse individually into adjacent layernorm/matmul operations.
- Register with `@torch.library.custom_op` and provide a functional impl Inductor can reason about.
- Add a custom Inductor pass that recognizes the steering op pattern and inlines it into adjacent ops.

**Expected impact**: eliminates the fixed per-step component. Hook count becomes nearly free (confirms the hook count ablation's linear scaling is fusion loss). ~0.5 ms/step savings across all batch sizes.

### 3. Batch `populate_steering_tables` into fewer CUDA calls

Currently launches ~100 small kernels per decode step (one `.zero_()` and up to one `.copy_()` per (layer, hook) combination). Instead: build a single flat update tensor on CPU, transfer once, scatter into tables with one call.

**Expected impact**: `populate_steering_tables` drops from ~2 ms/step in isolation to <0.5 ms/step. Most of the value at small batches. May also eliminate the "sync between populate and graph replay" candidate mechanism for #1 above.

### 4. Skip inactive hook points in populate

The loop always iterates all 3 hook points. When only 1 hook is actively set, 2/3 of the work is wasted zeroing rows. Track `active_hook_points: set[str]` and skip hook points outside the set.

**Expected impact**: ~60% populate-cost reduction when users set only one hook (the common case). Stacks with #3.

### 5. Cache populate output when config state is stable

Most decode steps don't change the active config set — same configs, same vectors. Add a "dirty" flag and skip populate when the state hasn't changed.

**Expected impact**: `populate_steering_tables` becomes effectively free in the steady state (most decode steps). Stacks with #3 and #4.

### 6. Pre-convert vectors at SamplingParams construction

`register_config` currently converts Python lists to CUDA tensors on every registration. Measurement: ~3 ms per register+release cycle at 1 config × 1 hook, scaling to ~145 ms at 16 configs × 3 hooks. Pre-converting at the serving boundary would move this cost out of the hot path.

**Expected impact**: eliminates a tail-latency risk at high-config-churn workloads. Not critical at current default `max_steering_configs=4`.

## Usage recommendations (with current implementation)

Pending the optimizations above, users can minimize steering overhead by:

1. **Use only one hook point** (`post_mlp` is usually sufficient). Adding hooks is linearly expensive until the fusion fix lands.
2. **Set `max_steering_configs` generously**: at minimum equal to the expected distinct-config concurrency, ideally with 2-4x headroom. Memory cost is trivial (~0.5 MB per config); latency cost of undersizing is severe (scheduler serializes requests).
3. **Mixed-workload deployments are fine.** Because per-batch cost is proportional to the number of active steering requests (not transitive), a single vLLM instance can serve mixed steered and non-steered traffic without penalizing the unsteered requests. A workload where only a small fraction of requests use steering pays cost proportional to that fraction.
4. **Prefer global steering vectors** (`collective_rpc("set_steering_vectors", ...)`) when all requests share the same steering target. Global vectors do not compete for table rows and do not trigger per-request registration overhead.

## Comparison to external libraries

Quantitative head-to-head comparison against TransformerLens, nnsight, repeng, and pyvene is **deferred to follow-up work** until the optimizations above are implemented. Comparing against a known-suboptimal version would mischaracterize vLLM's capability.

**Architectural advantages hold independently of the optimization work**:

- **Continuous batching with per-request vectors.** Other libraries either sequentialize multi-request workloads (TransformerLens, pyvene) or require the same intervention across a batch (repeng). vLLM deduplicates identical vectors and schedules distinct vectors in a single continuous batch.
- **Production serving stack.** OpenAI-compatible API, prefix caching, paged attention, KV cache reuse, multi-GPU support via tensor and pipeline parallel. Research libraries inherit none of this.
- **Zero cost to enable the feature.** `enabled_idle` mode shows ≤5% latency overhead across all batch sizes, so a team can deploy steering without penalty for requests that don't use it.
- **Scalable integration.** A team already running vLLM in production can add steering without adopting a new framework or reimplementing their serving pipeline.

## Follow-up benchmarks and profiling

### Open benchmarks

A **table sizing matrix** benchmark (`scripts/bench_table_sizing.py`) is available to isolate the interaction of `max_steering_configs × batch_size × distinct_configs` with two values of `max_steering_configs` (4 and 16), three distinct-config counts (1, 4, 8), and three batch sizes (8, 16, 32). Results for this sweep are pending a run; it replaces the ambiguity in the existing `ablation.config_scaling` benchmark (which conflates `max_steering_configs` with the number of distinct configs in its workload) by varying each axis independently.

The current `ablation.config_scaling` table above has a methodological caveat: its benchmark design couples `max_steering_configs` to `distinct_configs_in_workload` via `min(max_cfg, batch_size)`, so the 1→8 improvement cannot be cleanly attributed to row-count alone versus distinct-config-count alone. The plateau above the threshold is the clean, actionable finding from that table; the sub-threshold pattern is better isolated by the new `bench_table_sizing` benchmark.

### nsys / ncu profiling to distinguish fusion loss from per-active cost

The two-mechanism framing above rests on a linear-fit model that cannot mechanistically distinguish fusion loss from other per-active-request cost sources. `nsys` and `ncu` profiling would provide direct kernel-level evidence. Three experiments would pin down the mechanism:

**Experiment 1 — kernel list diff** (~15 min). Run `nsys profile` on a single batch-16 latency benchmark with and without steering. Compare kernel lists:

```bash
nsys profile --stats=true --output=trace_disabled.nsys-rep \
  uv run --no-sync python scripts/bench_latency.py \
    --batch-sizes 16 --iters 3 --warmup 1 --disable-prefix-cache --tag nsys-off
nsys profile --stats=true --output=trace_steering.nsys-rep \
  uv run --no-sync python scripts/bench_latency.py \
    --batch-sizes 16 --iters 3 --warmup 1 --tag nsys-on
nsys stats --report cuda_gpu_kern_sum trace_disabled.nsys-rep > kernels_disabled.txt
nsys stats --report cuda_gpu_kern_sum trace_steering.nsys-rep > kernels_steering.txt
diff kernels_disabled.txt kernels_steering.txt
```

Kernel names appearing only in the steering trace are unfused replacements. Kernel count delta and total GPU time delta directly measure fusion-loss magnitude.

**Experiment 2 — the decisive test** (~15 min). Run `nsys` on two configurations at the **same batch size** but different `num_active`: `(batch=16, num_active=4)` and `(batch=16, num_active=16)`. Compare per-step GPU kernel durations.

- If fusion loss is the dominant cost, the unfused kernels process all 16 tokens regardless of which are steered, so **GPU kernel time per step should be similar** between the two traces. Any wall-clock difference must be CPU-side (host-GPU syncs, dispatch overhead).
- If the cost is per-active and GPU-side, **kernel durations themselves should differ** — the steering-op calls (and any kernels whose cost depends on which tokens are active) would take longer at 16 active than at 4 active.

This experiment directly distinguishes Mechanism A from Mechanism B.

**Experiment 3 — NVTX section timing** (~10 min). Add NVTX annotations around the three suspected sections in `gpu_model_runner.py`:

```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("populate_steering_tables")
self._steering_manager.populate_steering_tables(self._steerable_layers)
nvtx.range_pop()

nvtx.range_push("build_steering_index")
# the per-request loop filling steering_index
...
nvtx.range_pop()

nvtx.range_push("model_forward")
result = self.model.forward(...)
nvtx.range_pop()
```

Then `nsys profile --capture-range=cudaProfilerApi` the run and open in `nsys-ui`. The NVTX timeline shows exactly which section's duration scales with `num_active`.

**Expected runtime for all three experiments: ~1 hour total.** The kernel-level evidence they produce would let the article make definitive claims about root cause rather than linear-fit-based hypotheses.

## Scaling projections

For a hypothetical dense 1T model (`hidden_size=20480, num_layers=160`):

- **Memory**: ~656 MB at `max_steering_configs=32`. Less than 0.03% of weights; negligible.
- **Per-step overhead (current implementation)**: extrapolating from Gemma-3-4B by `hidden_size × num_layers` scaling, per-step cost is estimated in the 1+ second range at batch=64 — **not viable** for production deployment without the optimizations above. The precise projection depends on whether the dominant `per-active-request` cost scales with `hidden_size × num_layers` (likely, if it's GPU memory bandwidth related) or stays roughly constant (unlikely, but possible if it's scheduler or engine bookkeeping).
- **Per-step overhead (post-optimization)**: if both root-cause mechanisms are addressed, the projected per-step overhead at batch=64 is in the low tens of milliseconds — making large-model steering feasible.

The optimization work is **not an incremental improvement but a requirement for scaling steering to production models**.
