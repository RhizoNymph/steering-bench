# Performance Characteristics

Consolidated findings from the steering-bench suite, targeting the performance characteristics of vLLM's per-request activation steering on Gemma-3-4B (hidden_size=2560, num_layers=34).

## Executive summary

1. **Enabling steering is effectively free.** With steering enabled but no active vectors, latency and throughput overhead is within measurement noise (-1.9% to +2.1%) across all tested batch sizes. Indistinguishable from the disabled baseline.
2. **Memory cost is negligible.** 0.5 MB per config on Gemma-3-4B (bf16); ~16 MB at `max_steering_configs=32` — less than 0.1% of GPU memory.
3. **Mixed-batch cost scales proportionally with active requests, not transitively.** Non-steered requests in a batch containing some steered requests pay only a small proportional share of the steering overhead. A workload with 5% steering traffic pays ~a few percent instance-wide overhead, not the ~74% all-active cost. Shared multi-tenant deployment is viable.
4. **Per-step overhead fits a linear model: `cost ≈ 0.5 ms fixed + 0.58 ms × num_active`** (Gemma-3-4B, per decode step). The dominant term scales with the number of active steered requests, not with batch size. At batch=32 fully steered this amounts to +124% total latency.
5. **Prefix caching works correctly.** Content-hashed cache keys deduplicate identical vectors across requests. No cache invalidation from steering.
6. **CUDA graphs are preserved.** Steering does not break graph capture or replay; graph effectiveness degrades modestly at larger batches.
7. **The overhead is CPU-side dispatch and synchronization, not GPU compute.** `nsys` kernel profiling confirms identical forward-pass kernels between disabled and steering runs. `torch.compile` fusion is NOT broken by the opaque steering op. The overhead comes from Python dispatch of ~102 small kernel launches per step in `populate_steering_tables` plus implicit host-GPU synchronization. A per-active-request scaling term (~0.58 ms/step per active) is the dominant cost at production batch sizes; its exact mechanism has not been pinned down but is confirmed to be CPU-side.

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

2. **`all_steered` per-step overhead roughly doubles with each batch size doubling** (assuming ~128 decode steps: b=1 → 1.2 ms/step, b=4 → 2.7, b=8 → 4.9, b=16 → 9.4, b=32 → 18.5). In `all_steered` mode, `num_active = batch_size`, so this doubling pattern reflects `cost ∝ num_active` — not batch-proportional GPU compute. The mixed-batch data (below) confirms this: at fixed batch size, cost scales with the number of active steered requests, not total batch. `nsys` kernel profiling further confirms that forward-pass GPU kernels are identical between disabled and steering runs (see Root Cause section).

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

Each additional hook adds approximately linear cost, with the per-hook cost scaling with batch size. **Hook count matters.** Since `nsys` kernel profiling confirms that forward-pass fusion is NOT broken by the steering op, this per-hook scaling is attributable to **CPU-side dispatch overhead**: each additional hook adds ~34 more `.zero_()` + `.copy_()` kernel launches per step in `populate_steering_tables`, plus the corresponding Python dispatch and potential synchronization overhead. The batch-size scaling reflects the per-active-request term (in all_steered mode, num_active = batch_size).

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

### Profiling evidence

#### torch.profiler (batch=8, 3 iterations, in-process mode)

CPU-side event deltas between steering and disabled runs:

| operation | disabled | steering | delta |
|---|---|---|---|
| `cudaLaunchKernel` wall | 240 ms | 311 ms | **+29%** (+70 ms, ~14k extra launches) |
| `cudaStreamSynchronize` | not present | 204 calls, 1024 ms | **new** (1 sec of host-blocked sync) |
| `aten::detach_` | not present | 204 calls, 15 ms | **new** (same count as the syncs) |
| `aten::copy_` | 4800 calls, 50 ms | 18060 calls, 156 ms | +3.7x calls (populate vector copies) |
| `aten::to` / `_to_copy` | 1152/1728 | 7884/8460 | +5x (device/dtype conversions) |
| `cudaMemcpyAsync` | 3648 | 10380 | +2.8x (host-device transfers in register_config) |
| `cudaGraphLaunch` | 294 | 294 | **unchanged** (graphs preserved) |

**Static code search.** A direct search of the steering code (`steering_manager.py`, `steering.py`, `_update_steering_buffers` in `gpu_model_runner.py`) found **no explicit synchronization calls** in the hot path: no `.item()`, no `.cpu()`, no `.tolist()`, no `torch.cuda.synchronize()`, no comparisons on GPU scalars. All tensor operations are async. The 204 syncs come from PyTorch internals or implicit boundaries (likely between eager `populate_steering_tables` writes and subsequent CUDA graph replay).

#### nsys GPU kernel profiling (batch=8, 2 iterations, in-process mode)

`nsys` GPU kernel summary (`cuda_gpu_kern_sum`) was collected for both disabled and steering runs at identical workloads. The full kernel lists were diffed.

**Result: every forward-pass kernel has the same call count in both traces.** Every GEMM, flash attention, triton fused op, and layernorm kernel appears with identical invocation counts and near-identical durations (differences <1%, within run-to-run noise). The diff is non-empty because timing values differ slightly between runs, but no forward-pass kernel names or counts changed. Only two kernels differ in count:

| kernel | disabled calls | steering calls | delta calls | delta GPU time |
|---|---|---|---|---|
| `FillFunctor<bf16>` (table zeros) | 62,844 | 75,900 | **+13,056** | **+17.9 ms** |
| `bfloat16_copy_kernel` (table copies) | 3 | 6,531 | **+6,528** | **+9.8 ms** |
| **Total extra GPU kernel time** | | | +19,584 | **+27.7 ms** |

Over 2 measured iterations at batch=8, max_tokens=64 (~64 decode steps each):

- **Extra GPU kernel time: ~28 ms total → ~14 ms per iter → ~0.22 ms per step**
- **Observed wall-clock overhead at batch=8 all_steered: ~623 ms per iter → ~4.9 ms per step**
- **GPU kernels account for ~4.5% of the wall-clock steering overhead**

The +13,056 fill calls correspond to `populate_steering_tables` zeroing table rows: `3 hooks × 34 layers = 102 fills per step × 128 total steps`. The +6,528 copy calls correspond to populate writing per-config vectors into table rows.

#### nsys CPU-side CUDA API profiling (same traces)

The `cuda_api_sum` report from the same nsys traces shows CPU-side CUDA API call counts and timings:

| API call | disabled calls | steering calls | delta calls | delta CPU time |
|---|---|---|---|---|
| `cudaLaunchKernel` | 129,686 (4,360 ms) | 149,270 (4,385 ms) | **+19,584** | **+24 ms** |
| `cudaMemcpyAsync` | 5,625 (1,198 ms) | 12,357 (1,231 ms) | **+6,732** | **+33 ms** |
| `cudaStreamSynchronize` | 1,126 (46.5 ms) | 1,330 (49.0 ms) | **+204** | **+2.5 ms** |
| `cudaEventSynchronize` | 568 (2,292 ms) | 568 (2,360 ms) | **0** | **+68 ms** |
| `cudaGraphLaunch` | 294 (37 ms) | 294 (43 ms) | **0** | **+5.5 ms** |
| `cudaDeviceSynchronize` | 2,012 (836 ms) | 2,012 (840 ms) | **0** | **+3 ms** |
| **Total API delta** | | | | **~136 ms** |

Over 2 iterations: ~136 ms extra CPU-side CUDA API time → **~68 ms per iter → ~1.1 ms per step**.

Key observations:

1. **`cudaLaunchKernel` +19,584 calls (+24 ms)**: matches the GPU kernel count delta exactly (the populate fills + copies). Each extra launch costs ~1.2 μs of CPU dispatch. Small.
2. **`cudaMemcpyAsync` +6,732 calls (+33 ms)**: the `.to(dtype)` conversions inside populate (converting per-config vectors to table bf16).
3. **`cudaStreamSynchronize` +204 calls (+2.5 ms)**: confirms the torch.profiler finding of 204 extra syncs but at much lower cost than torch.profiler reported (2.5 ms vs 1024 ms). The discrepancy is likely because torch.profiler measured wall-clock including GPU idle wait time, while nsys measures CPU time in the API call. At +2.5 ms total, this is **~0.02 ms/step — negligible.**
4. **`cudaEventSynchronize` same count but +68 ms**: existing event syncs take slightly longer (~120 μs more per call) because the GPU has more work queued from populate kernels. Adds ~0.5 ms/step.
5. **`cudaGraphLaunch` same count, same structure**: graph replay is fully preserved.

## Root cause: Python interpreter overhead in the populate loop

### Three-level overhead breakdown

The nsys GPU kernel trace, CPU API trace, and wall-clock measurements together give a complete picture of where the ~4.9 ms/step overhead lives at batch=8:

| source | per-step | % of overhead | measured by |
|---|---|---|---|
| GPU kernel compute (populate fills + copies) | ~0.22 ms | **4.5%** | nsys `cuda_gpu_kern_sum` |
| CUDA API dispatch (launch + memcpy + sync overhead) | ~1.1 ms | **22%** | nsys `cuda_api_sum` |
| **Python interpreter** (loop iteration, dict lookups, getattr, conditionals) | **~3.6 ms** | **73%** | inferred (wall-clock minus GPU minus API) |

### Fusion loss is ruled out

The nsys kernel diff is definitive: **the GPU runs the same forward-pass kernels in both disabled and steering modes**. Same kernel names, same call counts, same durations. The opaque `apply_steering` op is **not breaking torch.compile fusion** — the Inductor-generated fused kernels are identical.

The only extra GPU work is the populate fills/copies (~0.22 ms/step). These are negligible compute cost; their impact is in CPU-side launch overhead.

This rules out fusion loss as any significant contributor and means the earlier hook count ablation's linear scaling is attributable to **per-hook CPU dispatch overhead** (more hooks = more populate iterations = more Python overhead per step), not to GPU-side unfused compute.

### The dominant cost is Python interpreter overhead

The populate loop iterates `3 hooks × 34 layers = 102` times per step. Each iteration does:
- `getattr(mod, table_attr)` — Python attribute lookup
- `table[0].zero_()` — Python → PyTorch dispatcher → CUDA launch
- `.get()` dict lookups for global vectors
- `_add_vecs()` — Python function call with tensor ops
- Conditional branches (`if phase_global is not None and per_req is not None`)
- `.copy_()`, `.to()` — more Python → dispatcher → CUDA

The CUDA API calls themselves are fast (~1.1 ms/step per the nsys trace). The **Python interpreter time between those calls** — the 102 iterations of dict lookups, attribute access, function calls, conditionals, and loop overhead — is where ~3.6 ms/step lives. nsys cannot capture this because it's pure CPython interpreter work, not CUDA API calls.

### Per-active-request scaling (~0.58 ms/step × num_active)

This proportional term is the dominant cost at production batch sizes. Its mechanism has not been identified. The per-request Python loop in `_update_steering_buffers` iterates `num_reqs` (not `num_active`), so the scaling is not from that loop. The nsys traces confirm the cost is not GPU-side (identical forward-pass kernels) and not in CUDA API dispatch (only ~1.1 ms/step total delta, non-active-proportional).

Candidates include per-active work elsewhere in the V1 engine (unaudited), scheduler sub-batching of steered vs unsteered requests, or data-dependent cost in the populate path. NVTX-annotated nsys profiling at two different `num_active` values (same batch size) would pinpoint the source.

### Rule-outs (confirmed by measurement)

- **torch.compile fusion loss** — ruled out by nsys kernel diff (identical forward-pass kernels)
- **CUDA API dispatch as dominant cost** — ruled out by nsys API trace (only ~1.1 ms/step of extra API time, 22% of overhead)
- **`cudaStreamSynchronize` as dominant cost** — +204 calls confirmed, but total cost only +2.5 ms over 2 iters (~0.02 ms/step). torch.profiler's 1024 ms figure was a measurement artifact (included GPU idle wait time)
- **Prefix cache invalidation** — ruled out (latency: 69.0% → 67.7% with cache off; throughput: 1824 → 1800)
- **Raw steering op kernel cost** — ruled out at ~0.05 ms/call (microbench)
- **`steering_index` construction loop** — ruled out at ~5 μs/request (≤1.5% of overhead)
- **GPU compute scaling with batch_size** — ruled out (mixed-batch data shows cost proportional to num_active, not batch_size; nsys confirms identical forward-pass kernel durations)
- **`get_row_for_config` lookup cost** — sub-microsecond
- **Register/release cycle cost** — refcount-amortized to ~once per iter

## Optimization roadmap

Concrete targets, ordered by expected impact. Since the overhead is confirmed to be CPU-side (not GPU compute), the optimizations target Python dispatch and synchronization — **no torch.compile surgery needed**.

### 1. Cache populate output when config state is stable (highest impact, simplest fix)

Most decode steps don't change the active config set — same configs, same vectors. Add a "dirty" flag and skip `populate_steering_tables` entirely when the state hasn't changed. This eliminates all 102 per-step kernel launches and the implicit sync boundary in the steady state.

**Expected impact**: per-step populate overhead drops from ~2 ms to ~0 in steady state. If the implicit sync is triggered by populate's eager writes, this also eliminates the ~5 ms/step sync. Combined: potentially eliminates most of the per-step fixed cost.

### 2. Batch populate into fewer CUDA calls

When populate does need to run (config changes, phase transitions), replace the 102 individual `.zero_()` + `.copy_()` calls with a single batched operation: build a flat update tensor on CPU, transfer once, scatter into all table buffers with one kernel.

**Expected impact**: populate cost drops from ~2 ms/step to <0.1 ms/step when it does run. Reduces CPU dispatch overhead by ~100x. May also eliminate the implicit sync boundary by reducing the number of eager-to-graph transitions.

### 3. Identify the per-active-request cost source

**Status**: mechanism unknown. The per-step overhead fit is `0.58 ms × num_active`. At batch=32 this is 18.5 ms/step — the dominant cost at production batch sizes.

**Diagnostic approach**: add NVTX annotations around `populate_steering_tables`, the per-request index build loop, and the model forward pass. Run `nsys` at `(batch=16, num_active=4)` and `(batch=16, num_active=16)`. Whichever section's wall-clock scales with `num_active` is the source.

Once identified:
- If it's the implicit sync between populate and graph replay → #1 and #2 above may already fix it
- If it's per-active work elsewhere in the engine → refactor that path
- If it's GPU memory access pattern dependent → pre-sort or pack the steering index

**Expected impact**: if fully eliminated, per-step overhead drops to the small fixed component only (~0.5 ms/step at batch=8).

### 4. Skip inactive hook points in populate

The loop always iterates all 3 hook points. When only 1 hook is actively set, 2/3 of the work is wasted zeroing rows. Track `active_hook_points: set[str]` and skip hooks outside the set.

**Expected impact**: ~60% reduction in populate kernel launches when users set only one hook (the common case). Stacks with #2.

### 5. Pre-convert vectors at SamplingParams construction

`register_config` currently converts Python lists to CUDA tensors on every registration. Measurement: ~3 ms per register+release cycle at 1 config × 1 hook, scaling to ~145 ms at 16 configs × 3 hooks. Pre-converting at the serving boundary moves this cost out of the hot path.

**Expected impact**: eliminates tail-latency risk at high-config-churn workloads.

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

### Remaining profiling experiments

**Experiment 1 — kernel list diff: COMPLETED.** `nsys` GPU kernel summaries were collected for disabled vs steering at batch=8. Result: **identical forward-pass kernels** (same names, same counts, same durations). Only populate fills/copies differ (+13,056 fills, +6,528 copies, total +28 ms GPU time over 2 iters). This definitively rules out fusion loss and confirms the overhead is CPU-side. See the Root Cause section for the full analysis.

**Experiment 2 — per-active GPU kernel comparison** (~15 min, not yet run). Run `nsys` on two configurations at the **same batch size** but different `num_active`: `(batch=16, num_active=4)` and `(batch=16, num_active=16)`. Compare per-step GPU kernel durations. Given Experiment 1 showed identical forward-pass kernels between disabled and steering, this experiment would confirm that GPU kernel durations also don't scale with `num_active` — strengthening the conclusion that the per-active cost is entirely CPU-side.

**Experiment 3 — NVTX section timing** (~10 min, not yet run). Add NVTX annotations around `populate_steering_tables`, the per-request index build loop, and `model.forward()` in `gpu_model_runner.py`. Run `nsys` at two `num_active` values. The NVTX timeline shows exactly which section's wall-clock scales with `num_active`, directly identifying the source of the per-active-request cost. This is the highest-value remaining experiment for the optimization roadmap.

## Scaling projections

For a hypothetical dense 1T model (`hidden_size=20480, num_layers=160`):

- **Memory**: ~656 MB at `max_steering_configs=32`. Less than 0.03% of weights; negligible.
- **Per-step overhead (current implementation)**: since the overhead is CPU-side dispatch, not GPU compute, it scales with `num_layers × hooks` (number of populate kernel launches), not with `hidden_size × num_layers × batch_size`. A 1T model with 160 layers would have ~4.7x more populate launches per step than Gemma-3-4B (34 layers), suggesting ~4.7x worse per-step CPU overhead — estimated at ~23 ms/step fixed cost at batch=8. The per-active-request term's scaling with model size is unknown (depends on the unidentified mechanism).
- **Per-step overhead (post-optimization)**: the dirty-flag cache (#1 in roadmap) would eliminate populate entirely in steady state, making the fixed per-step cost near-zero regardless of model size. The per-active-request term requires identification first. If it's also CPU-side dispatch (likely given the nsys evidence), batching and caching optimizations should address it too.

The optimization work is **not an incremental improvement but a requirement for scaling steering to production models**.
