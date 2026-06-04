# Performance Characteristics

Consolidated findings from the steering-bench suite, targeting the performance characteristics of vLLM's per-request activation steering on Gemma-3-4B (hidden_size=2560, num_layers=34).

## Executive summary

1. **Enabling steering is effectively free.** With steering enabled but no active vectors, latency overhead averages **+0.5% across batch sizes 1–32** with individual measurements ranging from −2.1% to +5.4% (the +5.4% at b=4 is a noise outlier; the rest cluster within ±1%). Statistically indistinguishable from the disabled baseline.
2. **Memory cost is negligible.** 0.5 MB per config on Gemma-3-4B (bf16); ~16 MB at `max_steering_configs=32` — less than 0.1% of GPU memory.
3. **Mixed-batch cost scales proportionally with active requests, not transitively.** Non-steered requests in a batch containing some steered requests pay only a small proportional share of the steering overhead. A workload with 5% steering traffic pays ~a few percent instance-wide overhead, not the all-active cost. Shared multi-tenant deployment is viable.
4. **Steering overhead has two distinct components with different scaling characteristics:**
   - **Per-decode-step populate cost** (~2 ms/step in production, ~5 ms/step under nsys). Independent of `max_tokens`. Constant per step regardless of `num_active` in shared-vector mode (scales with `num_distinct_configs` only). The dirty-flag + batched-populate refactor (shipped) reduced this to near zero in steady-state decode steps.
   - **Per-request submission cost** (originally ~78 ms per steered request, now ~16 ms after the hash fix). One-time per `generate()` call. Amortizes over `max_tokens`, so it dominates at short outputs (`max_tokens ≤ 256`) and is negligible at long outputs (≥ 1k tokens).
5. **A previous version of this doc claimed `per_step_cost ≈ 0.7 ms + 0.57 ms × num_active`.** That model was *empirically correct* for the workloads measured (`max_tokens=128`) but *misinterpreted* — the "per-active per-step" term was actually a per-request submission cost amortized across decode steps. The number `0.57 ms/step × num_active` is `~73 ms/active ÷ 128 steps`. At `max_tokens=1024` the same submission cost would amortize to `~0.07 ms/step × num_active`; at `max_tokens=4096` to `~0.02 ms/step × num_active`. **The slope depends on output length, not on a per-decode-step mechanism.**
6. **Per-request submission cost: identified and fixed (the dominant win).** `hash_steering_config` was stringifying ~87,000 Python floats per request, twice per Request construction, taking ~60 ms. Plus `SamplingParams.clone()` was deep-copying the vectors per request for another ~17 ms. Three fixes (binary hash via `np.asarray(...).tobytes()`, `@cached_property` on `SamplingParams`, deepcopy memo for steering vector dicts) dropped per-request submission cost from ~78 ms → ~16 ms. Per-iter latency at b=16 fully steered dropped from 2989 ms → 1914 ms (−36%). Details in the [Per-request submission cost](#per-request-submission-cost-discovery-and-fix) section.
7. **Per-step populate cost: dirty flag + batched refactor (the follow-through).** After the hash fix, `populate_steering_tables` was the dominant remaining per-step cost. A `_tables_dirty` flag on `SteeringManager` skips populate entirely on steady-state decode steps (verified to run populate exactly **2 times per iter** instead of 128 at b=8). When populate does run, a new `index_copy_`-based write reduces kernel launches vs the old per-row copies. End-to-end improvement was modest (~2.5-4% additional vs post-hash-fix) because most of populate's CPU work was already overlapped with GPU forward-pass compute — an important lesson about isolated microbenchmark time vs critical-path wall clock. **At `max_tokens=2048` with 16 steered requests at b=16, per-step latency is statistically identical to the disabled baseline** — steering at long output lengths is essentially free. The refactor also surfaced a subtle `torch.tensor(list, device='cuda')` synchronization trap that had made the first batched implementation 2.4× slower per call than the per-row baseline. Details in the [Populate refactor](#populate-refactor-shipped) section.
8. **Prefix caching works correctly.** Content-hashed cache keys deduplicate identical vectors across requests. No cache invalidation from steering.
9. **CUDA graphs are preserved.** Steering does not break graph capture or replay; graph effectiveness degrades modestly at larger batches.
10. **The remaining overhead is CPU-side launch dispatch, not GPU compute.** `nsys` kernel profiling confirms identical forward-pass kernels between disabled and steering runs. `torch.compile` fusion is NOT broken by the opaque steering op. The ~2 ms of remaining per-populate-call cost is ~400 tiny CUDA kernel launches with ~5 μs of CPU dispatch each — launch-overhead bound, not compute-bound. Further optimization would require collapsing those launches (single contiguous backing tensor or a custom Triton kernel).

## Methodology

- **Hardware**: Single NVIDIA RTX 3090 (24 GB, 936 GB/s HBM)
- **Model**: `google/gemma-3-4b-it` (bf16)
- **Steering vectors**: Random, seeded, scale=0.1, hook point = `post_mlp` unless noted otherwise
- **Measurement**: Wall-clock via `time.perf_counter` with `torch.cuda.synchronize()` barriers for vLLM benchmarks; CUDA events for microbenchmarks
- **Prefix caching**: Enabled by default unless explicitly disabled in ablation
- **Iterations**: Microbenchmarks use 100 warmup / 1000 measured; system benchmarks use 5 warmup / 20 measured
- **Runs pinned to same machine** with no concurrent GPU workloads

### Note on data collection

The end-to-end benchmark numbers below were re-collected after a fix to vLLM's gpu_model_runner that eliminated **phantom populate overhead in disabled mode**. The original `gemma3.py` model registers steering buffers unconditionally, so even with `enable_steering=False` the engine was running the populate path with no active configs every step (~1.5 ms/step of pure dispatch overhead). The fix adds an early-return when `vllm_config.steering_config is None`, plus a no-active-state short-circuit that skips `populate_steering_tables` when no per-request configs are active and the global state is unchanged. After the fix:
- The `disabled` baseline drops by ~5–15% across batch sizes (no longer paying for phantom populate)
- The `enabled_idle` mode is now equivalent to `disabled` (the short-circuit catches it)
- Steering-active runs are unchanged (the populate path still runs with active configs)
- Pre-fix benchmark records have been moved to `results-archive/pre-fix/` and are not used in any table below

The profiling subsections (`torch.profiler`, `nsys`) were collected against the pre-fix vLLM. The kernel-diff conclusions still hold (forward-pass kernels are identical between disabled and steering — the fix only removed populate work from the disabled side, making the diff *cleaner*, not different in character). Absolute kernel counts in those tables should be read as "disabled paid for the bug".

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
| `disabled` | 1499 | 1459 | 1495 | 1569 | 1842 |
| `enabled_idle` | 1502 | 1538 | 1494 | 1564 | 1803 |
| `mixed_25` | — | 1733 | 1771 | 1979 | 2502 |
| `mixed_50` | — | 1736 | 1905 | 2247 | 3078 |
| `mixed_75` | — | 1804 | 2039 | 2542 | 3647 |
| `all_steered` | 1704 | 1872 | 2176 | 2830 | 4259 |

Throughput (tokens/sec):

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `disabled` | 128 | 526 | 1027 | 1958 | 3340 |
| `enabled_idle` | 128 | 500 | 1028 | 1964 | 3408 |
| `mixed_25` | — | 443 | 867 | 1552 | 2456 |
| `mixed_50` | — | 442 | 806 | 1367 | 1996 |
| `mixed_75` | — | 426 | 753 | 1208 | 1685 |
| `all_steered` | 113 | 410 | 706 | 1085 | 1443 |

Latency overhead (%) vs `disabled` at same batch size:

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `enabled_idle` | +0.2% | +5.4% | -0.1% | -0.3% | -2.1% |
| `mixed_25` | — | +18.7% | +18.4% | +26.1% | +35.8% |
| `mixed_50` | — | +19.0% | +27.4% | +43.2% | +67.1% |
| `mixed_75` | — | +23.6% | +36.4% | +62.0% | +98.0% |
| `all_steered` | +13.6% | +28.3% | +45.6% | +80.4% | +131.2% |

Throughput loss (%) vs `disabled` at same batch size:

| mode | b=1 | b=4 | b=8 | b=16 | b=32 |
|---|---|---|---|---|---|
| `enabled_idle` | -0.2% | -5.0% | +0.1% | +0.3% | +2.0% |
| `mixed_25` | — | -15.8% | -15.6% | -20.7% | -26.5% |
| `mixed_50` | — | -15.9% | -21.5% | -30.2% | -40.2% |
| `mixed_75` | — | -19.1% | -26.7% | -38.3% | -49.6% |
| `all_steered` | -12.0% | -22.0% | -31.3% | -44.6% | -56.8% |

Three key observations from this table:

1. **`enabled_idle` is statistically indistinguishable from `disabled`.** Overheads range from −2.1% to +5.4% (mean +0.5%), with signs distributed across positive and negative. Since `enabled_idle` cannot mechanically be faster than `disabled`, the negative values confirm the signal is zero and the variance is measurement noise. The +5.4% at b=4 is a single outlier (1538 vs 1459 ms); the other four batch sizes are all within ±0.3%. Enabling the steering feature without active vectors has no measurable cost.

2. **`all_steered` per-step overhead roughly doubles with each batch size doubling** (assuming ~128 decode steps: b=1 → 1.6 ms/step, b=4 → 3.2, b=8 → 5.3, b=16 → 9.9, b=32 → 18.9). In `all_steered` mode, `num_active = batch_size`, so this doubling pattern reflects `cost ∝ num_active` — not batch-proportional GPU compute. The mixed-batch data (below) confirms this: at fixed batch size, cost scales with the number of active steered requests, not total batch. `nsys` kernel profiling further confirms that forward-pass GPU kernels are identical between disabled and steering runs (see Root Cause section).

3. **At batch=32 fully steered, steering more than doubles total latency** (+131.2%). Per-request deployment without the optimizations described below is not viable at production batch sizes.

4. **Per-request latency overhead converges with batch size.** Computing `total_overhead / num_active` for the `all_steered` rows: b=1 → 205 ms/request, b=4 → 103, b=8 → 85, b=16 → 79, b=32 → 76. This asymptotically approaches ~76 ms per active request as the batch grows, consistent with the per-step linear model amortizing a small fixed per-step cost across more active requests at larger batches.

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

> **⚠ Correction (2026-04-13).** This section originally interpreted the per-active slope as a per-decode-step cost. It is not. The slope is a **per-request submission cost** that gets amortized over `max_tokens` when expressed per-step. The numbers in the tables below are correct as *per-iter, per-active* costs but the original "per-step linear model" framing was misleading. See the [Per-request submission cost](#per-request-submission-cost-discovery-and-fix) section for the full investigation. The numbers below also reflect *pre-fix* state — after the hash fix, the per-active slope drops from ~76 ms/active to ~16 ms/active.

Per-active-request marginal latency cost, computed from the throughput matrix as `(latency_with_n_active - latency_disabled) / n`:

| batch size | all_steered cost/active |
|---|---|
| 4 | 103 ms |
| 8 | 85 ms |
| 16 | 79 ms |
| 32 | 76 ms |

Cost per additional active request **converges to ~76 ms** at larger batch sizes and stays stable whether the active fraction is 25%, 50%, 75%, or 100%. Checked directly by computing successive deltas at batch=32 across the four mode points (mixed_25, mixed_50, mixed_75, all_steered correspond to num_active = 8, 16, 24, 32):

- 0 → 8 active: +660 ms ÷ 8 = **82.5 ms/active**
- 8 → 16 active: +576 ms ÷ 8 = **72.0 ms/active**
- 16 → 24 active: +569 ms ÷ 8 = **71.1 ms/active**
- 24 → 32 active: +612 ms ÷ 8 = **76.5 ms/active**

The b=16 dedicated mixed-batch benchmark sweeps `num_active = 1, 4, 8, 16` at fixed batch_size=16 and yields a slope of **74.8 ms per additional active request** from a 4-point linear fit, matching the throughput-matrix deltas above and confirming the slope is intrinsic to `num_active` rather than an artifact of mode binning.

**Per-step "linear model" — corrected interpretation.** The original fit was:

`per_step_cost ≈ 0.7 ms fixed + 0.57 ms × num_active`

This is empirically correct *for the workload it was measured on* (`max_tokens=128`), but the second term is **not actually a per-decode-step cost**. The 0.57 ms/step "per active" is `~73 ms per request submission ÷ 128 decode steps`. At a different `max_tokens`, the apparent per-step slope would be different:

| max_tokens | apparent per-active slope (post-divide) |
|---|---|
| 64 | ~1.14 ms/step × num_active |
| 128 | ~0.57 ms/step × num_active |
| 256 | ~0.28 ms/step × num_active |
| 512 | ~0.14 ms/step × num_active |
| 1024 | ~0.07 ms/step × num_active |
| 2048 | ~0.04 ms/step × num_active |

The fixed term (`~0.7 ms fixed` per step) IS a real per-decode-step cost — it's the populate-loop work. The per-active term is *not*. See the [Per-request submission cost](#per-request-submission-cost-discovery-and-fix) section for how this was confirmed via NVTX experiments and what the actual root cause turned out to be.

**Per-active per-iter slope (the real number):**

Per-batch-size fit quality check (slopes from the dedicated `bench_mixed_batch` runs, which sweep `num_active` at constant `batch_size`):

| batch | per-active slope (ms/iter) | per-active slope (ms/step) |
|---|---|---|
| 4  | 75.9 | 0.59 |
| 8  | 67.3 | 0.53 |
| 16 | 74.8 | 0.58 |

The slope is **invariant across batch sizes** — exactly what the linear cost model predicts. Per-active marginal cost is a property of the workload, not of the batch shape. A second independent measurement comes from the cuda_graphs ablation, where the per-step overhead at `num_active = batch_size` divided by `num_active` gives 0.59 (b=4), 0.64 (b=8), 0.61 (b=16), 0.59 (b=32), 0.59 (b=64) ms/step per active — same slope, two unrelated benchmarks, five batch sizes.

**Crucially, there is no large batch-proportional term.** If fusion loss were the dominant cost, the per-active slope should *not* stay flat across batch sizes — fusion loss would manifest as cost scaling with total batch_size even at constant num_active, producing a clear batch-proportional contribution on top of the per-active term. The fits do not show this.

**Deployment implication:** A vLLM instance with `enable_steering=True` serving mixed traffic pays cost proportional to actual steering usage. For a workload where 5% of requests use steering, the effective instance-wide latency overhead is a few percent, not the ~76% all-active cost. Shared multi-tenant inference with steering is viable without penalizing non-steering users.

### Ablations

**CUDA graphs (enforce_eager × enable_steering, batch sweep 1–64)**

| batch | graphs_no_steer (ms) | graphs_w_steer (ms) | steer_overhead (w/ graphs) | graph_speedup (no steer) | graph_speedup (w/ steer) |
|---|---|---|---|---|---|
| 1 | 1503 | 1742 | +15.9% | 2.92x | 2.79x |
| 4 | 1477 | 1876 | +27.0% | 3.10x | 2.76x |
| 8 | 1537 | 2192 | +42.6% | 3.00x | 2.49x |
| 16 | 1609 | 2846 | +76.9% | 2.82x | 2.14x |
| 32 | 1846 | 4278 | +131.7% | 2.53x | 1.72x |
| 64 | 1974 | 6744 | +241.6% | 2.34x | 1.45x |

CUDA graph benefit is **preserved at small batches** (ratio 0.95 at b=1) and **degraded at large batches** (ratio 0.62 at b=64). Graphs still replay correctly; the cost is compute inside the graph, not graph overhead.

**Per-step steering overhead** (derived from cuda_graphs ablation, dividing by 128 decode steps):
- batch=1: ~1.9 ms/step
- batch=4: ~3.1 ms/step
- batch=8: ~5.1 ms/step
- batch=16: ~9.7 ms/step
- batch=32: ~19.0 ms/step
- batch=64: **~37.3 ms/step**

Per-step overhead scales approximately linearly with batch size in this all-steered configuration. Dividing per-step overhead by `batch_size` (= num_active here) gives 1.87 → 0.78 → 0.64 → 0.61 → 0.59 → 0.58 ms/step per active request — converging to the same ~0.57 ms/step slope identified in the mixed-batch fits.

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

> **Note on pre-fix collection.** The torch.profiler and nsys traces below were captured against the pre-fix vLLM (with phantom populate in disabled mode). The kernel-diff conclusion — "forward-pass kernels are identical between disabled and steering" — is **strengthened** by the fix, not weakened: the fix only removes work from the disabled side, so the post-fix delta would show *more* extra populate kernels in the steering trace, not different ones. The wall-clock per-step overhead figures (~4.9 ms/step at b=8) match the post-fix cuda_graphs measurements above (5.1 ms/step at b=8) within noise, so the cited percentages and conclusions stand.

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

> **Note (2026-04-13).** This section was the original analysis of the per-step overhead breakdown at b=8. It is **still valid for the per-decode-step component of overhead**, but it predates the discovery of the per-request submission cost. The "~5.1 ms/step at batch=8" figure includes both the per-step populate cost (the focus of this section) AND amortized per-request submission cost (which is a separate problem, now mostly fixed). For a complete picture, read this section together with [Per-request submission cost: discovery and fix](#per-request-submission-cost-discovery-and-fix) below.

### Three-level overhead breakdown

The nsys GPU kernel trace, CPU API trace, and wall-clock measurements together give a complete picture of where the ~5.1 ms/step overhead at batch=8 lives (post-fix wall-clock from cuda_graphs ablation; nsys GPU and API deltas are pre-fix but sum the same populate work). **Note**: this measurement was at `max_tokens=128`, so it includes amortized submission cost. The genuine per-decode-step components are the GPU kernel and CUDA API rows; the "Python interpreter" row mixes per-step populate work with the per-request submission cost we later identified.

| source | per-step | % of overhead | measured by |
|---|---|---|---|
| GPU kernel compute (populate fills + copies) | ~0.22 ms | **4%** | nsys `cuda_gpu_kern_sum` |
| CUDA API dispatch (launch + memcpy + sync overhead) | ~1.1 ms | **22%** | nsys `cuda_api_sum` |
| **Python interpreter** (loop iteration, dict lookups, getattr, conditionals) — *includes amortized per-request submission cost* | **~3.8 ms** | **74%** | inferred (wall-clock minus GPU minus API) |

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

### Per-active-request scaling — IDENTIFIED (was originally listed as "mechanism unknown")

The original version of this doc listed the per-active term as "~0.57 ms/step × num_active, mechanism unknown". After NVTX-instrumented `nsys` profiling, the mechanism was identified: **the cost lives in `LLM._add_completion_requests` (request submission), not in the per-decode-step path**. The "0.57 ms/step" was an artifact of dividing a per-request submission cost by `max_tokens=128`. Root cause: `hash_steering_config` stringifying ~87,000 floats per request. See [Per-request submission cost: discovery and fix](#per-request-submission-cost-discovery-and-fix) below for the full investigation and the three-part fix that dropped it by 81%.

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

## Per-request submission cost: discovery and fix

This section documents the investigation that revealed the "per-active scaling" was not actually a per-decode-step cost, and the three-part fix that dropped per-request submission cost by ~80%.

### Why the original "per-step linear model" was misleading

The throughput-matrix and mixed-batch benchmarks all measure per-iter wall clock and divide by `max_tokens` to get a per-step number. If a steered request has *any* one-time cost that scales with `num_active` and runs once per `generate()` call, that cost gets amortized across decode steps and looks like a per-step cost in the resulting tables. With `max_tokens=128` and a 76 ms-per-active per-iter cost, the "per-step" number was `76 / 128 ≈ 0.58 ms/step × num_active`. We initially attributed this to "per-active engine work in the per-decode-step path" but never identified which code was actually doing the work.

### The investigation

To pin down the source we used **NVTX-instrumented `nsys` profiling** with shared steering vectors (so `num_distinct_configs = 1` regardless of `num_active`, isolating any per-active cost from populate's inner loop). Two runs at the same `batch_size=16` but different `num_active` (4 and 16) — the linear model predicted a per-step delta of `~6.84 ms/step`. We instrumented every section of the per-decode-step path:

1. **First experiment** — NVTX ranges around `_update_steering_buffers`, `_update_states`, `_prepare_inputs`, `_model_forward`, `_preprocess`, `_build_attention_metadata`, the postprocess block. **Result: every range was flat across n=4 → n=16. Total per-step delta in instrumented code: ~50 μs/step out of an expected 6840 μs/step.** The per-step path had essentially no per-active scaling.
2. **Second experiment** — added NVTX inside `LLMEngine.add_request`, `EngineCore.step_with_batch_queue`, scheduler calls, `sample_tokens`, `_bookkeeping_sync`, output assembly. **Result: still flat.** The per-step path was confirmed clean.
3. **Third experiment** — added NVTX in `_render_and_add_requests` (with a per-call tag distinguishing steered vs unsteered submissions), `LLM._add_request`, `LLMEngine.add_request`'s sub-calls, and the InprocClient path. **Found it:** `_render_and_add_requests:steered` median was ~78.5 ms per request, while `_render_and_add_requests:plain` was ~0.11 ms per request — a 700× per-call cost difference for steered vs unsteered submissions. The per-active "scaling" lived entirely in the request submission path that runs **once per `generate()` call before the engine loop ever starts**.

### Decomposing the 78 ms per-request cost

Drilling down within `_add_request`, the breakdown was:

| range | per-call median | % of submission |
|---|---|---|
| `input_processor.process_inputs` | ~17 ms | 22% |
| `engine_core.add_request` → `preprocess_add_request` | ~62 ms | 80% |
| `EngineCore.add_request_inner` (the actual scheduler enqueue) | ~5 μs | ~0% |
| everything else | negligible | — |

The 62 ms in `preprocess_add_request` was traced to `Request.from_engine_core_request` → `Request.__init__`, which accesses two `@cached_property` hashes (prefill + decode) that call `hash_steering_config(effective_*_steering)`. The 17 ms in `input_processor.process_inputs` was traced to `params.clone()`, which does `copy.deepcopy(self)` on the SamplingParams and recursively traverses the steering vector dict.

### Root cause: `hash_steering_config` stringified 87,000 floats

The original implementation:

```python
def hash_steering_config(effective_vectors):
    if not effective_vectors:
        return 0
    canonical = {hp: sorted(vecs.items()) for hp, vecs in sorted(effective_vectors.items())}
    data = str(sorted(canonical.items())).encode()   # <-- str() on 87k floats
    return int(hashlib.sha256(data).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
```

The `str(sorted(canonical.items()))` line stringifies every Python float in the steering vector dict. For Gemma-3-4B at 1 hook, that's `34 layers × 2560 floats = 87,040 floats`. Each `float.__repr__` produces a 16-character string (`"0.123456789012345"`) at ~1–3 μs per float. Total stringification: **~28 ms per call**. Plus the SHA-256 over the resulting ~2-3 MB string. Called twice per Request (prefill + decode hashes). Total: ~62 ms per Request, matching the measured `preprocess_add_request` cost exactly.

A microbench confirmed: **`hash_steering_config` took 27.6 ms/call** before the fix.

### The three-part fix

**1. Binary hash function** (`vllm/config/steering_types.py`): replace `str()` with `np.asarray(vec, dtype=np.float32).tobytes()` and feed binary bytes directly into `hashlib.sha256.update`. `tobytes()` is essentially a memcpy and SHA-256 on bytes is hardware-accelerated.

```python
def hash_steering_config(effective_vectors):
    if not effective_vectors:
        return 0
    h = hashlib.sha256()
    for hook in sorted(effective_vectors.keys()):
        h.update(hook.encode())
        layer_dict = effective_vectors[hook]
        for layer_idx in sorted(layer_dict.keys()):
            entry = layer_dict[layer_idx]
            if isinstance(entry, dict):
                vec = entry.get("vector", entry)
                scale = float(entry.get("scale", 1.0))
            else:
                vec = entry
                scale = 1.0
            arr = np.asarray(vec, dtype=np.float32)
            h.update(layer_idx.to_bytes(4, "little", signed=True))
            h.update(arr.tobytes())
            if scale != 1.0:
                h.update(np.float64(scale).tobytes())
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
```

**Microbench result: 27.6 ms → 1.9 ms per call (14.5× faster).**

**2. Cache the hash on `SamplingParams`** (`vllm/sampling_params.py`): add `@cached_property` for `prefill_steering_config_hash` and `decode_steering_config_hash`. Many requests sharing the same `SamplingParams` instance — the common case for `llm.generate(prompts, [sp]*N)` — only compute the hash once across the whole batch instead of once per request. `Request.prefill_steering_config_hash` (and the decode equivalent) now delegate to `sampling_params.prefill_steering_config_hash` instead of computing fresh.

**3. Fix `SamplingParams.clone()`** (`vllm/sampling_params.py`): `process_inputs` clones SamplingParams per request to mutate `max_tokens` and apply generation-config / tokenizer overrides. The clone was using `copy.deepcopy(self)` which recursively traversed the entire steering vector dict (~87K floats nested in dicts and lists), costing ~17 ms per request. Two changes:
   - **Pre-populate the deepcopy memo** with the top-level steering vector dicts mapped to themselves. `copy.deepcopy` checks the memo before recursing, so any object found there is returned by reference instead of deep-copied. The clone shares the steering vector dicts with the original by reference (safe because nothing downstream mutates them).
   - **Explicitly carry over cached `@cached_property` values** (the prefill/decode hashes plus the `effective_*_steering` resolutions) from `self.__dict__` to `new_sp.__dict__`, since msgspec.Struct's deepcopy path doesn't reliably preserve these.

### Measured impact

Path B test (no nsys overhead, b=16, shared vector, max_tokens=128, mean of 10 measured iters):

| metric | pre-fix | hash + cache | hash + cache + clone | total improvement |
|---|---|---|---|---|
| n=4 mean (ms/iter) | 1989.2 | 1773.2 | **1727.4** | **−262 ms (−13%)** |
| n=16 mean (ms/iter) | 2988.9 | 2113.1 | **1914.6** | **−1074 ms (−36%)** |
| per-iter delta n=16 vs n=4 | 999.7 | 339.9 | **187.2** | **−812 ms (−81%)** |
| per added active request | **83.3 ms** | 28.3 ms | **15.6 ms** | **−68 ms (−81%)** |

Per-step latency at b=16 fully steered:
- Pre-fix: 23.4 ms/step
- Post-fix: **15.0 ms/step**
- Disabled baseline: ~12.3 ms/step
- **Per-step overhead vs disabled dropped from ~11 ms/step → ~2.7 ms/step (a ~75% reduction)**

NVTX confirmation (post-Option-1, pre-Option-3): `preprocess_add_request` median per call dropped from ~62 ms → 8.6 ms (−86%), exactly matching the predicted hash-fix savings. After the clone fix the residual in `process_inputs` should drop further.

### What's left in the residual ~16 ms per request

After all three fixes, ~16 ms per steered request submission remains. From the NVTX breakdown, it splits roughly as:
- **~5 ms in `input_processor.process_inputs`** — non-deepcopy work (validation, EngineCoreRequest construction)
- **~8 ms in `preprocess_add_request`** — Request construction (block hashes, scheduling state)
- **~3 ms in other per-request bookkeeping** — output_processor.add_request, assign_request_id, etc.

These don't have an obvious single-fix winning solution and amortize away at long output lengths (at `max_tokens=1024`, 16 ms/request is 0.0125 ms/step — well below noise). They're below the threshold for further investigation.

### Implications for the deployment story

The two-cost decomposition matters because the costs have **different scaling characteristics with output length**:

- **Per-decode-step populate cost** (~5 ms/step under nsys, ~2 ms/step in production): independent of `max_tokens`. For a chatbot generating 1024 tokens, this is ~2 sec total per request — the dominant overhead for long-output workloads. Eliminable by the populate refactor (item #1 in the optimization roadmap).
- **Per-request submission cost** (~16 ms/request post-fix, was ~78 ms pre-fix): one-time per `generate()` call. For a 1024-token request that's `16/1024 ≈ 0.016 ms/step` — completely negligible at long outputs. The hash fix already lands this at a usable level; the populate refactor is the next priority.

For short-output workloads (`max_tokens ≤ 256`, e.g. classification, scoring), the post-fix submission cost still dominates, but at 16 ms per request it's now small enough that even short-output workloads see acceptable overhead. **Pre-fix, a `max_tokens=64` workload at b=16 fully steered was paying `(78 × 16)/64 = 19.5 ms/step` of pure submission tax. Post-fix: `(16 × 16)/64 = 4 ms/step`.**

## Populate refactor (shipped)

After the hash fix, the dominant remaining per-step overhead was `populate_steering_tables` — the function that writes the latest steering-vector contents into each layer's per-hook table buffer before every forward pass. Two optimizations shipped together:

### Part 1: dirty flag (items #1 in the optimization roadmap above)

Most decode steps don't change the active config set. Only three events actually mutate the steering state: a new per-request config is registered (`register_config` new-row path), the last reference to an existing config is released (`release_config` refcount→0 path), or a global vector is updated / cleared. Every other decode step finds populate regenerating identical table contents.

Added `_tables_dirty: bool` to `SteeringManager`, set by every state mutator and cleared at the end of `populate_steering_tables`. `_update_steering_buffers` now checks the flag before calling populate.

**Verified behavior**: instrumented `populate_steering_tables` with a call counter and ran `llm.generate()` at b=8 all_steered with a shared vector. Before the dirty flag: **128 calls per iter** (one per decode step). After: **2 calls per iter** (one for initial registration, one for the prefill→decode phase transition). The other ~126 calls are short-circuited entirely.

**But the end-to-end savings were smaller than expected.** The per-iter wall clock at b=8 all_steered dropped from 1.98 ms/step (post-hash-fix) to 1.34 ms/step (with dirty flag) — a 32% reduction in the *remaining* overhead, but only ~0.65 ms/step savings × 128 steps ≈ 83 ms per iter. Why so small when populate was supposedly 2.5 ms/step?

**Because most of populate's cost was already overlapped with GPU forward-pass compute.** An isolated microbenchmark (with `torch.cuda.synchronize()` around each populate call) shows populate really does take ~2.76 ms of wall clock in isolation. But inside the engine pipeline, populate's CPU dispatch work happens while the GPU is still chewing through the previous step's forward pass — most of the 2.76 ms is absorbed into GPU busy time. Only the ~0.65 ms tail, where populate's last dispatches block the next forward pass from starting, shows up as measured wall-clock improvement.

This is an important subtlety about the earlier `populate_steering_tables | 2.50 ms | 49% of per-step overhead` claim in the [Three-level overhead breakdown](#three-level-overhead-breakdown) section above — that 2.50 ms was the function's isolated CPU time, but only ~0.65 ms of it was actually on the critical path.

### Part 2: batched populate with a debugging detour (item #2)

When populate does need to run (the two times per iter that survive the dirty-flag check), the original implementation issued `(3 + num_active_configs)` separate `.zero_()` / `.copy_()` calls per `(hook, layer)` iteration. At 3 hooks × 34 layers on Gemma-3-4B, that's ~408 tiny CUDA kernel launches per populate call.

Refactored the inner `(hook, layer)` body into a helper `_populate_one_table` that:
1. Builds a Python list of the row-content tensors (`[zero_row, global_prefill_effective, global_decode_effective, *per_request_rows]`)
2. `torch.stack`s the list into a dense `(num_rows, hidden_size)` float32 tensor
3. Dtype-converts once via `.to(table.dtype)`
4. Writes to the per-layer table buffer with a single `table.index_copy_(0, indices, stacked)`

**The first version was 2.4× slower per call than the original.** A microbenchmark (200 populate calls at b=8, shared vector) showed 2.76 ms/call (per-row) vs **6.55 ms/call** (batched). Not the expected direction.

**Root cause** (found via nsys CPU API diff): the inner loop was calling
```python
indices = torch.tensor(target_indices, dtype=torch.long, device=table.device)
zero_row = torch.zeros(hidden_size, dtype=torch.float32, device=table.device)
```
per `(hook, layer)` iteration, 102 times per populate call. `torch.tensor(list, device='cuda')` does a synchronous host→device copy, and `torch.zeros(..., device='cuda')` triggers a small allocation + fill kernel launch. Over 200 calls in the microbenchmark, the batched variant was doing **~20,000 extra `cudaMemcpyAsync` calls** with matching `cudaStreamSynchronize` waits — adding up to **~2 seconds** of pure synchronization overhead that the per-row version didn't have. The nsys API diff was unambiguous:

| API call | per-row | batched (broken) | batched (fixed) |
|---|---|---|---|
| `cudaLaunchKernel` | 149,270 / 4.41s | 149,270 / 4.45s | 129,878 / 4.35s |
| `cudaMemcpyAsync` | 12,357 / 1.26s | **25,413** / 1.31s | 6,021 / 1.21s |
| `cudaStreamSynchronize` | 1,330 / 49ms | **20,914 / 2.00s** | 1,522 / 1.78s |

**The fix**: hoist both `indices` and `zero_row` out of the `(hook, layer)` loop. They're identical across every iteration within a single populate call (the row ordering depends only on `config_to_row` state, and the scratch zero vector has no per-layer content), so we can build them once per call outside the loop. The loop body passes them into `_populate_one_table` as arguments.

After the fix:
- Microbench per-call: **2.78 ms/call** (matching per-row within noise)
- GPU kernel time per call: **~40% less** than per-row (62 ms vs 105 ms over 200 calls)
- ~20,000 fewer CPU-side API calls per trace

**Takeaway**: in a system that's dominated by launch overhead rather than GPU compute (populate issues ~400 tiny kernels per call, each doing ~2 KB of work), *where the allocations happen matters more than how much data moves*. `torch.tensor(list, device='cuda')` is a latent synchronization point that doesn't show up in the code until you profile. The moral: measure your "batched" refactor against the per-row baseline with wall-clock timing — if the batched version is slower, the issue is almost always the setup code, not the batched write itself.

### Measured end-to-end impact

Throughput matrix at `max_tokens=128`, comparing `postfix-hash` (hash fix only) to `postfix-populate` (hash fix + dirty flag + batched populate), all `all_steered`:

| batch | postfix-hash (ms/iter) | postfix-populate (ms/iter) | delta | overhead vs disabled (post-populate) |
|---|---|---|---|---|
| 1 | 1590 | 1550 | −40 (−2.5%) | +3.3% |
| 4 | 1621 | 1555 | −66 (−4.1%) | +6.7% |
| 8 | 1722 | 1657 | −65 (−3.8%) | +10.9% |
| 16 | 1963 | 1900 | −64 (−3.2%) | +21.0% |
| 32 | 2545 | 2481 | −64 (−2.5%) | +34.6% |

The populate refactor shaves ~60 ms/iter off the all_steered workload across every batch size — a ~2.5-4% additional improvement on top of the hash fix. Not dramatic in percentage terms, but internally consistent (the ~60 ms is what the dirty flag saves by avoiding ~126 populate calls × ~0.5 ms each of critical-path work).

**The big story is the max_tokens sweep**, which shows that at long output lengths the overhead effectively **vanishes**:

| max_tokens | n=0 per-step (ms) | n=16 per-step (ms) | overhead at n=16 |
|---|---|---|---|
| 64 | 12.27 | 17.49 | +42.5% |
| 128 | 12.73 | 14.91 | +17.1% |
| 256 | 12.78 | 13.75 | +7.6% |
| 512 | 13.29 | 13.61 | +2.4% |
| **1024** | **14.22** | **14.12** | **−0.7%** (within noise) |
| **2048** | **14.86** | **14.85** | **−0.0%** (identical to idle baseline) |

At `max_tokens=2048`, per-step latency with 16 steered requests is **statistically identical** to per-step latency with no active steering configs. The per-request submission cost has fully amortized away and the per-step populate cost has been eliminated by the dirty flag — **steering at long output lengths is essentially free**.

### Why the populate refactor isn't a bigger percentage win

Both parts shipped, both are measurably correct, but the end-to-end improvement vs the post-hash-fix baseline is modest in percentage terms (~3% at b=16). The reason, as the [dirty flag analysis](#part-1-dirty-flag-items-1-in-the-optimization-roadmap-above) above explains, is that **most of populate's CPU dispatch work was already being overlapped with GPU forward-pass compute** before the refactor. The critical path through a decode step was never dominated by populate; it was dominated by forward pass GPU time, which populate happened to run in parallel with.

For the article's headline number, **the hash fix was the dominant win** — it removed ~60 ms per request of actual critical-path work. The populate refactor is a correctness-minded cleanup that validates the two-cost decomposition (per-step + per-request) and demonstrates the "isolated microbenchmark CPU time ≠ critical-path wall clock" gap that's easy to conflate.

### What's next: custom Triton kernel (future work)

The remaining populate-attributable cost is ~400 kernel launches per call, each dispatching a tiny `(1, hidden_size)` copy or similar. At ~5 μs of CPU dispatch overhead per launch, that's ~2 ms of pure Python→CUDA dispatch per call — overwhelmingly the dominant cost, since the GPU compute per call is <1 ms. To get further improvements you'd need to:

1. **Single contiguous backing tensor** across all `(hook, layer)` table buffers — collapses the ~400 launches to 1-3, but requires changes to the per-layer buffer registration path and CUDA graph capture timing. Estimated 4-6 hours of work for a ~0.3% end-to-end improvement. Not pursued.
2. **Custom Triton kernel** that takes the config-vector backing store as input and scatters to all layer-hook table slots in a single launch. Would eliminate populate's launch overhead almost entirely. Requires learning Triton and handling the memory-view mapping. Estimated 1-2 days. Planned as follow-up.

## Optimization roadmap

Concrete targets, ordered by expected impact. Since the overhead is confirmed to be CPU-side (not GPU compute), the optimizations target Python dispatch and synchronization — **no torch.compile surgery needed**.

**Status as of 2026-04-13**: items #1, #2, and #3 are all **shipped**. Item #3 (per-request submission cost) delivered the dominant end-to-end win. Items #1 and #2 (populate refactor) landed afterward and delivered a modest additional win plus an unexpected debugging lesson — see the [Populate refactor](#populate-refactor-shipped) section below. The remaining optimization candidates (items #4, #5, and custom Triton kernel for populate) are still open.

### 1. Cache populate output when config state is stable (✅ shipped — dirty flag)

Most decode steps don't change the active config set — same configs, same vectors. A `_tables_dirty: bool` flag on `SteeringManager` is set by every state mutator (`register_config` new-row, `release_config` refcount→0, `update_global_vectors`, `clear_global_vectors`) and cleared at the end of `populate_steering_tables`. `_update_steering_buffers` checks the flag before calling populate.

**Measured impact** (shared-vector workload at b=8 all_steered, verified directly by counting populate calls per iter): populate runs exactly **2 times per iter** instead of 128 (once for the initial config registration, once for the prefill→decode phase transition). The other ~126 calls are skipped entirely.

**End-to-end impact** is much smaller than the 126×-skip ratio suggests because most of populate's wall-clock cost was already overlapping with GPU forward-pass compute — the CPU-side kernel launches happen during GPU busy time, so eliminating them only recovers the small tail that was on the critical path. Measured: per-step overhead at b=8 all_steered dropped from 1.98 ms/step (post-hash) to 1.34 ms/step (post-populate) — a 32% reduction of the *remaining* overhead, or ~0.65 ms/step saved per populate call avoided.

**Files changed**: `vllm/v1/worker/steering_manager.py` (flag + mutator sets + clear), `vllm/v1/worker/gpu_model_runner.py` (pre-populate check in `_update_steering_buffers`).

### 2. Batch populate into fewer CUDA calls (✅ shipped — with a debugging detour)

When populate does run, the original implementation issued `(3 + num_active_configs)` small kernel launches per `(hook, layer)` iteration — one `.zero_()`/`.copy_()` per row. Refactored to assemble the row contents into a single Python list, `torch.stack` into a dense tensor, dtype-convert once, and scatter into the table buffer with a single `index_copy_` per `(hook, layer)`.

**The debugging story** (see also [Populate refactor](#populate-refactor-shipped) below): the first version of this refactor was **2.4× slower** in isolated microbenchmark per call than the code it replaced. Root cause: `torch.tensor(list, device='cuda')` and `torch.zeros(..., device='cuda')` were being called per-`(hook, layer)` iteration inside the inner loop, each triggering a synchronous host→device copy. Over a short benchmark run this added up to **~2 seconds** of `cudaStreamSynchronize` that the per-row version didn't have. The fix: hoist both the `indices` tensor and the `zero_row` scratch out of the inner loop — build them once per populate call from the already-known `config_to_row` snapshot.

**Measured impact after the fix**: per-call populate wall-clock is within 1% of the per-row baseline (2.78 vs 2.81 ms), uses ~40% less GPU kernel time (~62 ms vs ~105 ms over 200 calls), and issues ~20,000 fewer CPU-side API calls per trace. The end-to-end benefit stacks with the dirty flag from item #1.

**Files changed**: `vllm/v1/worker/steering_manager.py` (new `_populate_one_table` helper + hoisted scratch in `populate_steering_tables`).

### 3. Per-request submission cost (✅ shipped — hash function rewrite + per-SP caching + clone deepcopy memo)

**Status**: identified, root-caused, and fixed. See the [Per-request submission cost](#per-request-submission-cost-discovery-and-fix) section above for the full investigation.

**Summary of fix**: `hash_steering_config` was stringifying ~87,000 Python floats per request via `str(sorted(canonical.items()))`. Three changes:
1. Replace `str()` with `np.asarray(...).tobytes()` (binary hash). 27.6 ms → 1.9 ms per call.
2. Cache the hash on `SamplingParams` as `@cached_property`. Many requests sharing one SP only hash once.
3. Fix `SamplingParams.clone()` to share steering vectors by reference via the deepcopy memo, and explicitly carry over cached hash values.

**Measured impact**: per-steered-request submission cost dropped from ~83 ms → ~16 ms (5.3× faster). Per-iter latency at b=16 fully steered dropped 36%. Per-step overhead vs disabled dropped from ~11 ms/step → ~2.7 ms/step (−75%).

**Files changed**:
- `vllm/config/steering_types.py` — `hash_steering_config` rewrite
- `vllm/sampling_params.py` — `@cached_property` for the hashes; `clone()` deepcopy memo + cache propagation
- `vllm/v1/request.py` — `Request.prefill_steering_config_hash` and `decode_steering_config_hash` now delegate to `SamplingParams` cached values

### 4. Skip inactive hook points in populate

The loop always iterates all 3 hook points. When only 1 hook is actively set, 2/3 of the work is wasted zeroing rows. Track `active_hook_points: set[str]` and skip hooks outside the set.

**Expected impact**: ~60% reduction in populate kernel launches when users set only one hook (the common case). Stacks with #2.

### 5. Pre-convert vectors at SamplingParams construction

`register_config` currently converts Python lists to CUDA tensors on every registration. Measurement: ~3 ms per register+release cycle at 1 config × 1 hook, scaling to ~145 ms at 16 configs × 3 hooks. Pre-converting at the serving boundary moves this cost out of the hot path.

**Expected impact**: eliminates tail-latency risk at high-config-churn workloads.

## Usage recommendations (post hash fix)

With the hash fix shipped, the per-request submission cost is no longer a meaningful overhead in most workloads. The main remaining bottleneck is the per-decode-step populate cost, which the populate refactor (items #1 and #2 above) will address.

1. **Use only one hook point** (`post_mlp` is usually sufficient). Adding hooks is linearly expensive in the populate path until the dirty-flag + batched-populate refactor lands.
2. **Set `max_steering_configs` generously**: at minimum equal to the expected distinct-config concurrency, ideally with 2-4x headroom. Memory cost is trivial (~0.5 MB per config); latency cost of undersizing is severe (scheduler serializes requests).
3. **Mixed-workload deployments are fine.** Because per-batch cost is proportional to the number of active steering requests (not transitive), a single vLLM instance can serve mixed steered and non-steered traffic without penalizing the unsteered requests. A workload where only a small fraction of requests use steering pays cost proportional to that fraction.
4. **Prefer global steering vectors** (`collective_rpc("set_steering_vectors", ...)`) when all requests share the same steering target. Global vectors do not compete for table rows and do not trigger per-request registration overhead.
5. **Long output lengths help.** The per-request submission cost is a one-time per-`generate()` cost. At `max_tokens=64` it shows up as ~0.25 ms/step per active; at `max_tokens=2048` it's ~0.008 ms/step per active. Workloads that generate longer outputs see proportionally smaller per-step overhead from this cost. With the hash fix, even short-output workloads (`max_tokens=64`) see acceptable overhead (~4 ms/step submission tax at b=16 fully steered, vs 19.5 ms/step pre-fix).
6. **Reuse `SamplingParams` objects across requests in a batch.** The hash fix's `@cached_property` cache lives on the SamplingParams instance, so when the user passes `[sp_steered]*N` (the common pattern for shared steering targets), the hash is computed exactly once across the whole batch instead of N times. This is automatic in `bench_throughput_matrix` and `bench_mixed_batch`; if you're constructing a fresh SamplingParams per request, you'd lose the caching benefit.

## Comparison to external libraries

Quantitative head-to-head comparison against TransformerLens, nnsight, repeng, and pyvene is **deferred to follow-up work** until the optimizations above are implemented. Comparing against a known-suboptimal version would mischaracterize vLLM's capability.

**Architectural advantages hold independently of the optimization work**:

- **Continuous batching with per-request vectors.** Other libraries either sequentialize multi-request workloads (TransformerLens, pyvene) or require the same intervention across a batch (repeng). vLLM deduplicates identical vectors and schedules distinct vectors in a single continuous batch.
- **Production serving stack.** OpenAI-compatible API, prefix caching, paged attention, KV cache reuse, multi-GPU support via tensor and pipeline parallel. Research libraries inherit none of this.
- **Zero cost to enable the feature.** `enabled_idle` mode is statistically indistinguishable from `disabled` (mean +0.5% latency overhead across batch sizes 1–32, with sign distribution consistent with measurement noise), so a team can deploy steering without penalty for requests that don't use it.
- **Scalable integration.** A team already running vLLM in production can add steering without adopting a new framework or reimplementing their serving pipeline.

## Follow-up benchmarks and profiling

### Open benchmarks

A **table sizing matrix** benchmark (`scripts/bench_table_sizing.py`) is available to isolate the interaction of `max_steering_configs × batch_size × distinct_configs` with two values of `max_steering_configs` (4 and 16), three distinct-config counts (1, 4, 8), and three batch sizes (8, 16, 32). Results for this sweep are pending a run; it replaces the ambiguity in the existing `ablation.config_scaling` benchmark (which conflates `max_steering_configs` with the number of distinct configs in its workload) by varying each axis independently.

The current `ablation.config_scaling` table above has a methodological caveat: its benchmark design couples `max_steering_configs` to `distinct_configs_in_workload` via `min(max_cfg, batch_size)`, so the 1→8 improvement cannot be cleanly attributed to row-count alone versus distinct-config-count alone. The plateau above the threshold is the clean, actionable finding from that table; the sub-threshold pattern is better isolated by the new `bench_table_sizing` benchmark.

### Remaining profiling experiments

**Experiment 1 — kernel list diff: COMPLETED.** `nsys` GPU kernel summaries were collected for disabled vs steering at batch=8. Result: **identical forward-pass kernels** (same names, same counts, same durations). Only populate fills/copies differ (+13,056 fills, +6,528 copies, total +28 ms GPU time over 2 iters). This definitively rules out fusion loss and confirms the overhead is CPU-side. See the Root Cause section for the full analysis.

**Experiment 2 — per-active GPU kernel comparison** (~15 min, not yet run). Run `nsys` on two configurations at the **same batch size** but different `num_active`: `(batch=16, num_active=4)` and `(batch=16, num_active=16)`. Compare per-step GPU kernel durations. Given Experiment 1 showed identical forward-pass kernels between disabled and steering, this experiment would confirm that GPU kernel durations also don't scale with `num_active` — strengthening the conclusion that the per-active cost is entirely CPU-side.

**Experiment 3 — NVTX section timing: COMPLETED.** Multiple rounds of NVTX-instrumented `nsys` profiling were run with shared steering vectors at b=16 with `num_active=4` and `num_active=16`. Instrumentation eventually covered every section of `execute_model`, `sample_tokens`, `EngineCore.step_with_batch_queue`, `LLMEngine.step`, `LLMEngine.add_request`, `_render_and_add_requests`, and `_add_request`. Result: **the per-active scaling is entirely in `LLM._add_completion_requests`** (request submission), not in the per-decode-step path. Root-caused to `hash_steering_config` calling `str()` on 87K Python floats per request. See the [Per-request submission cost](#per-request-submission-cost-discovery-and-fix) section above for the full investigation and the three-part fix that resolved it.

## Scaling projections

For a hypothetical dense 1T model (`hidden_size=20480, num_layers=160`):

- **Memory**: ~656 MB at `max_steering_configs=32`. Less than 0.03% of weights; negligible.
- **Per-step populate overhead (current implementation)**: since the overhead is CPU-side dispatch, not GPU compute, it scales with `num_layers × hooks` (number of populate kernel launches), not with `hidden_size × num_layers × batch_size`. A 1T model with 160 layers would have ~4.7x more populate launches per step than Gemma-3-4B (34 layers), suggesting ~4.7x worse per-step CPU overhead — estimated at ~23 ms/step fixed cost at batch=8. This is the dominant remaining overhead for the populate refactor (items #1 and #2) to address.
- **Per-step populate overhead (post-populate-refactor)**: the dirty-flag cache (#1 in roadmap) would eliminate populate entirely in steady state, making the fixed per-step cost near-zero regardless of model size.
- **Per-request submission overhead (post hash fix)**: the binary hash function scales as `num_layers × hidden_size` per call. For a 1T model with 160 layers × 20480 floats = 3.3M floats, the binary hash should take ~30-50 ms per call (vs ~2 ms for Gemma-3-4B). With the `@cached_property` cache on `SamplingParams`, this is paid exactly once per `generate()` call regardless of how many requests share the SP, so it amortizes to negligible at any reasonable `max_tokens`. For per-request distinct vectors, it's a 30-50 ms one-time submission cost per request — still small relative to the per-decode-step populate cost on a model that big.

The optimization work is **not an incremental improvement but a requirement for scaling steering to production models**. With the hash fix shipped, the per-decode-step populate cost is now the binding constraint.

## Fix history

- **2026-04-09**: Initial benchmark suite produces "0.5 ms/step + 0.58 ms/step × num_active" linear cost model from `bench_throughput_matrix` and `bench_mixed_batch`. Per-active mechanism listed as "unknown but CPU-side".
- **2026-04-10**: Phantom populate fix lands in vLLM. `gemma3.py` was registering steering buffers unconditionally so even `enable_steering=False` was running populate every step. Fix: early-return in `gpu_model_runner` when `vllm_config.steering_config is None`, plus a no-active-state short-circuit. Disabled baseline drops 5–15% across batch sizes; pre-fix benchmark records archived to `results-archive/pre-fix/`.
- **2026-04-12**: NVTX investigation begins. NVTX ranges added throughout `execute_model`, `sample_tokens`, `EngineCore.step_with_batch_queue`, `LLMEngine.step`. Surprise finding: the per-step path has no per-active growth in shared-vector mode. Cost lives outside the per-step path entirely. Investigation continues into `LLMEngine.add_request` and `LLM._add_completion_requests`.
- **2026-04-13 (morning)**: Root cause of per-request submission cost identified — `hash_steering_config` stringifies 87K Python floats per call, called twice per Request from `Request.__init__`. Microbench confirms 27.6 ms/call. Three-part fix lands:
  - Binary hash function (`vllm/config/steering_types.py`) — 27.6 ms → 1.9 ms per call.
  - `@cached_property` on `SamplingParams` for the prefill/decode hashes — many requests sharing one SP only hash once.
  - `SamplingParams.clone()` deepcopy memo + cache propagation (`vllm/sampling_params.py`) — eliminates the per-request 17 ms deepcopy of 87K floats and ensures the hash cache survives the clone.
  - Combined impact: per-iter wall clock at b=16 fully steered drops from 2989 ms → 1914 ms (−36%). Per-active scaling drops from 83 ms → 16 ms per added request (−81%). Per-step overhead vs disabled drops from ~11 ms/step → ~2.7 ms/step (−75%).
  - Commit: `perf(steering): make hash_steering_config 14x faster and cache it`
- **2026-04-13 (afternoon)**: Populate refactor lands.
  - **Dirty flag** (`vllm/v1/worker/steering_manager.py` + `vllm/v1/worker/gpu_model_runner.py`): `_tables_dirty` flag on `SteeringManager`, set by every state mutator and cleared at the end of `populate_steering_tables`. `_update_steering_buffers` checks the flag before calling populate. Verified by instrumentation: populate runs exactly **2 times per iter** instead of 128 at b=8 all_steered (initial registration + prefill→decode transition). Commit: `perf(steering): skip populate when state is unchanged via dirty flag`
  - **Batched populate via `index_copy_`** (`vllm/v1/worker/steering_manager.py`): inner `(hook, layer)` body refactored to stack row contents and write with a single scatter per layer-hook pair. First implementation was 2.4× slower per call than the per-row version it replaced, because `torch.tensor(list, device='cuda')` and `torch.zeros(..., device='cuda')` called per iteration triggered ~20,000 synchronous host→device copies (`cudaMemcpyAsync` + `cudaStreamSynchronize`) per benchmark run — ~2 seconds of pure sync waits. Fixed by hoisting both scratch tensors out of the inner loop and building them once per call. After fix: microbench wall-clock matches per-row within 1%, ~40% less GPU kernel time, ~20,000 fewer CPU-side API calls per trace. Commit: `perf(steering): batch per-layer table writes via index_copy_`
  - **End-to-end impact**: ~2.5-4% additional latency reduction vs post-hash-fix across all batch sizes (64 ms/iter saved at b=16 all_steered: 1963 → 1900). Smaller than expected because most of populate's CPU work was already overlapping with GPU forward-pass compute in the pipeline — only the ~0.5 ms critical-path tail per populate call became wall-clock savings. **The interesting end state**: at `max_tokens=2048`, per-step overhead at b=16 with 16 steered requests is indistinguishable from the disabled baseline (−0.0% at n=16). Steering at long output lengths is essentially free. See the [Populate refactor](#populate-refactor-shipped) section for the full analysis.
- **Pending follow-up**: custom Triton kernel for populate. The remaining populate cost is ~400 tiny CUDA kernel launches per call (launch-overhead bound, not GPU-compute bound). Collapsing them into a single custom kernel would eliminate most of the remaining populate cost. Estimated 1-2 days of work. Not pursued yet.
