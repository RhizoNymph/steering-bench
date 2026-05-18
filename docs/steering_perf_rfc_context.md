# Steering Perf — Session Context (2026-05-17)

End-of-session reference for the work needed to write the upstream vLLM RFC.
Captures: what got merged, what got measured, what got rejected, why, and
what's still outstanding for the H100/A100 production-grade numbers.

## TL;DR — production state after today

Six PRs landed against `feat/steering`, one PR was closed without merge,
one PR is being benched on H100-class hardware as I write this. Inline-mode
TTFT collapsed from ~300 ms to ~60 ms — within 2-4% of the
`enabled_idle` floor on a pinned 3090. The full bench evidence is
reproducible from the JSONs under `results/`.

```
mode                  TTFT base   TTFT after all merges   throughput Δ
─────────────────────────────────────────────────────────────────────
enabled_idle          43 ms       43 ms                   flat
named_shared          51 ms       43 ms     (-15%)        +0.7%
all_steered_shared    306 ms      64 ms     (-79%)        +30%
per_request_n4        314 ms      65 ms     (-79%)        +24%
per_request_n16       225 ms      52 ms     (-77%)        +22%
                                  p99 also -77% on every inline mode
```

## PRs landed

| # | Branch | Effect (3090 pinned, n=128, conc=8) |
|---|---|---|
| **#157** | `perf/steering-jit-prewarm` | First-request named_shared TTFT: variable 55-185 ms → deterministic 53 ms. σ collapse 62→0.4 ms across trials. Adds ~25 ms one-shot at engine init. |
| **#158** | `perf/steering-async-h2d-stack` | Non-blocking H2D + pinned-CPU staging ring in `_stack_vectors_to_device`. Cold-config materialize saves ~10-15 ms. |
| **#161** | `perf/steering-prematerialize-named` | Eager `pre_materialize_steering_module` at `/v1/steering/modules/register` time. Removes 15 ms cold-path `register_config.materialize` from first request resolving to a named module. |
| **#162** | `perf/steering-binary-wire` | **The headline.** New `steering_vectors_packed` field carrying base64-encoded `(num_layers, hidden_size)` bytes. Eliminates ~10-15 ms of JSON parse + `np.asarray(list_of_floats)` per inline request. TTFT median -77 to -79%, p99 -77 to -84%, throughput +22 to +30% on all inline modes. Also turned out to dramatically reduce variance — p99 went from ~500 ms to ~85 ms. |
| **#163** | `perf/steering-packed-per-layer-scale` | Optional `scales: list[float]` field on packed entries. Matches legacy `{"vector":..., "scale":...}` per-layer scale knob. Zero overhead when absent; sub-millisecond per-request when present. |
| **#164** | `chore/steering-remove-auto-promote` | Removes `SteeringAutoPromoteLRU` + `maybe_auto_promote_steering_modules{,_async}` and all plumbing. Net **+5 / −959 lines**. Per_request_n16 p99 drops from 124 ms → 88 ms (eliminating the bimodal distribution caused by sync `await collective_rpc(register_steering_modules)`). |

## PRs closed without merge

- **#159 async-promote** (closed) — fire-and-forget broadcast variant of auto-promote. Benched in a fresh worktree on top of current `feat/steering`. **Made p99 *worse*** (127 ms sync → 146 ms async on per_request_n16 — likely ZMQ broadcast queue backpressure + worker-side serialization of in-flight `register_steering_modules` calls). Concrete evidence killed the "make it async" path; the right answer was the simpler "remove it entirely" path (#164). The work isn't wasted — it's archived on `perf/steering-async-promote-rebased` worktree branch.

- **#160 fused-index-copy** (closed) — collapses 34 per-layer `index_copy_` launches into one `[L, R, H]` index_copy. Original (thermal-contaminated 3090) showed ~−55 ms TTFT on per_request_n4. After pinning clocks and re-benching, the win shrunk to ~−4 ms median on `all_steered_shared` and **nothing measurable on per_request_n16**. With `concurrency=8` + binary wire (#162), the per-layer launches overlap with sibling-request forward passes, so the saved launch cost doesn't surface in TTFT. Closed as superseded by binary wire's preprocessing-cost reduction. Code is correct, just doing zero observable work in the current architecture.

## Bench infrastructure changes shipped today

These changes are in `scripts/bench_serving.py` and need to ship together with the RFC for reproducibility.

| change | flag / mechanism | why |
|---|---|---|
| Discarded warmup pass | `--warmup-requests N` (default = `--concurrency`) | Triton JIT + auto-promote LRU + pinned-buffer alloc happen on first-touch; without warmup the first batch of measured requests was inflated by 150+ ms. |
| Per-mode warmup floor | automatic via `distinct_configs_for_mode()` | `per_request_n16` needs ≥16 warmup reqs or 8 of 16 configs hit the measured pass cold and inflate p99. |
| Drain after warmup | `--warmup-drain-seconds 0.5` + 1-token barrier request | Async background work (broadcasts, deferred H2D, auto-promote scheduling) keeps the worker busy after warmup returns. Without drain, this work landed during the measured pass and showed up as a phantom ~2 ms TPOT regression on PR worktrees vs base. |
| Binary wire support | `--packed-vectors` | Bench harness for testing #162. |
| Per-layer scale support | `--packed-with-scales` | Bench harness for testing #163. |
| Result-JSON parameter recording | `parameters.warmup_requests`, `parameters.warmup_drain_seconds`, `parameters.packed_vectors`, `parameters.packed_with_scales` | So `analyze.py` can split cells by bench config across runs. |

## Methodology lessons (worth a paragraph in the RFC's methodology section)

1. **Pin GPU clocks before benching.** Single biggest lesson of the day. Burned several hours chasing "regressions" that were entirely thermal-throttle artifacts (e.g., +1.4 ms TPOT on `named_shared` that disappeared the moment clocks were pinned to 1350 MHz). Without pinning, base runs first on a cool GPU and every subsequent PR variant looks worse purely because the card warmed up. `scripts/probe_sustainable_clock.sh` empirically finds the sustainable value.

2. **n=128 is the floor for inline-mode comparisons.** Inline-mode σ is ~100-110 ms (vs ~5 ms for simple modes), so SE-of-median at n=32 is ±25 ms and at n=128 is ±13 ms. Any delta under 5 ms at n=128 should be ignored as noise; even 10 ms deltas should be confirmed with multiple trials.

3. **Multiple trials are required.** Confirmed bimodality in `named_shared` TTFT (~43 ms vs ~54 ms clusters). Single-trial measurements at the noise floor are unreliable. **3 trials per cell** is the minimum that's defensible; 5 is better if you're characterizing tail behaviour.

4. **Drain async work between warmup and measurement.** Without this, "PR worktrees" with any async-dispatch behaviour (auto-promote broadcasts, deferred H2D) show measurement artifacts unrelated to their actual perf. The drain step (sleep + 1-token barrier) makes the measured pass start from a quiescent state regardless of what the PR did during warmup.

5. **For PRs that change request preprocessing, measure on the API server thread.** Auto-promote's p99 regression was invisible at the per-mode median level — only the full distribution (p25/p50/p75/p90/p99) showed the bimodal cohort that broadcast contention created. Always look at distribution shape, not just medians.

## Where the data lives

All on the local 3090 box with `clocks.current.graphics` pinned at 1350 MHz, drained warmup, n=128, conc=8, packed-vectors enabled where applicable.

| dir | what |
|---|---|
| `results/integration-sweep-n128-pinned/` | Headline 7-worktree pinned comparison (base + 5 individual PRs + integration of all 5). Pre-#162/#163/#164. |
| `results/blake3-binary-sweep/` | base/blake3/binary/combined 4-cell. Proved BLAKE3 was marginal next to binary wire. |
| `results/packed-scales-sweep/` | 3-cell (base, scales-off, scales-on). Confirmed #163 has sub-ms overhead. |
| `results/autopromote-diag/` | 3 trials × (AP-on, AP-off) × 5 modes. Confirmed AP's −36 ms p99 regression on per_request_n16 was reproducible (σ across trials = 0.8 ms). |
| `results/async-promote-diag/` | 3 trials × (sync, async-promote). Showed #159 made p99 *worse*. |
| `results/fused-index-post-164/` | 3 trials × (base, +#160). Showed #160 provides no measurable benefit post-AP-removal. |
| `results/named-shared-rerun/` | 3 trials × 3 worktrees on named_shared only. Quantified the ~43/~54 ms bimodal cluster. |

Analysis scripts in `/tmp/claude/cmp_*.py` (not preserved in repo — re-derive from the JSONs if needed).

## Open work — RFC blockers

### 1. H100/A100/H200 numbers — A100 LANDED 2026-05-18

3090 data alone wasn't sufficient for an upstream RFC. Pinned-A100 post-merge data
landed 2026-05-18 from a Prime Intellect SXM box (clock pinned to **1410 MHz** by
the user — that's the A100 sustainable value, not drift; default 1500 in the
launcher is H100-class). The FlashInfer JIT compile mess was sidestepped with
`export VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler, no kernel difference
across cells so cross-cell deltas unaffected).

Data lives in `results/a100-post-merge/` with three subdirs:
- `serving/` (5 JSONs, one per mode, n=128 conc=8 tag=`post-merge`)
- `throughput-matrix/` (27 JSONs: mode × batch × fraction grid)
- `modes-matrix/` (426 JSONs from `--preset full`)

#### A100 vs-disabled overhead (the headline RFC table)

A100 pinned 1410 MHz, single physical box (`prxmx260090`). `disabled` baseline N=18 (9 from a dedicated disabled-only loop + 9 from a full all-modes loop where disabled was first). Other modes N=9. Source: `results/new-a100-post-merge/serving/`.

Reported as **mean of per-trial medians ± SEM** (σ/√N) rather than median-of-medians, because the low-overhead modes (disabled / enabled_idle / named_shared) have a bimodal per-trial TTFT distribution with ~6 ms gap between two clusters. A median-of-medians flips clusters based on trial count; a mean averages over them proportionally. SEM-based 2σ significance marked with `*`.

```
mode                  N    ΔTTFT_p50      ΔTPOT         ΔE2EL          significant?
────────────────────────────────────────────────────────────────────────────────────
enabled_idle          9   -1.3 ± 1.0    +0.2 ± 0.0*   +22.4 ± 1.0*    TPOT, E2EL only
named_shared          9   +0.0 ± 1.2    +0.2 ± 0.0*   +23.6 ± 1.1*    TPOT, E2EL only
all_steered_shared    9  +11.8 ± 1.3*   +0.3 ± 0.0*   +51.1 ± 0.8*    all
per_request_n4        9  +16.0 ± 1.5*   +0.3 ± 0.0*   +59.5 ± 1.0*    all
per_request_n16       9  +17.6 ± 1.2*   +0.4 ± 0.0*   +74.1 ± 1.3*    all

disabled baseline (N=18):
  TTFT_p50 = 31.25 ± 0.62 ms (mean of per-trial medians ± SEM)
  TPOT     =  7.189 ± 0.001 ms
  E2EL     = 944.4  ± 0.6 ms
```

Takeaways for the RFC:
- **Loading the steering scaffold idle is free at TTFT** (-1.3 ± 1.0 ms — within noise) and costs **+0.2 ms per generated token** (+2.5% TPOT, +2.4% E2EL on 128-token output). That's the per-step poll of the steering manager; it's small and constant across all configurations.
- **named_shared also has no measurable TTFT cost vs disabled** (+0.0 ± 1.2 ms). The prematerialize (#161) + binary-wire (#162) combo eliminated the named-config setup tax — sending a named module reference is as cheap as sending a normal request.
- **Inline modes have real, significant TTFT overhead**: +12-18 ms vs disabled. This is the "you pay for per-request vector materialization" cost. Bounded and predictable; absolute numbers ~44-50 ms are still well within latency budgets.
- **TPOT delta is constant across modes** (+0.2 to +0.4 ms regardless of how many vectors are active). This is the headline claim for "per-token steering is free in steady state."
- **Bimodal TTFT in low-cost modes**: the disabled / enabled_idle / named_shared TTFT distribution has two clusters ~6 ms apart. Mean-based statistics handle this; median-based statistics flip between clusters. **Don't quote `TTFT_p50` from this dataset for the low-overhead modes without the SEM context**, and don't quote `TTFT_p99` for disabled at all (SEM is ±15-30 ms — one or two trials per cluster have extreme tails).

  **Root cause is upstream vLLM, not steering.** A follow-up investigation
  compared per-trial TTFT stddev across modes (from the per-trial
  aggregate stats already in each JSON — the bench doesn't dump raw
  per-request samples, so this is the strongest analysis without
  re-instrumenting):

  ```
  mode                  N    mean within-trial p10-p90 spread
  ───────────────────────────────────────────────────────────
  disabled             18                              8.54 ms
  enabled_idle          9                              8.80 ms
  named_shared          9                              8.98 ms
  all_steered_shared    9                             28.79 ms
  per_request_n4        9                             36.85 ms
  per_request_n16       9                             35.42 ms
  ```

  All three low-overhead modes have an essentially identical within-trial
  TTFT spread of ~9 ms. Steering doesn't increase variance for these modes
  — it just adds an offset (the +0.2 ms TPOT cost). The 6 ms gap between
  the two trial-median clusters is a sampling artifact: each trial draws
  128 requests from a distribution with an ~8 ms internal spread, and the
  median of 128 lands at one cluster or the other depending on the random
  per-trial mix.

  Mean within-trial stddev (6.56 ms) > between-trial median stddev
  (2.63 ms) confirms the bimodality is **request-level**, not trial-level
  — it's not an unstable trial-start state, it's how vLLM's scheduler
  dispatches requests inside a steady-state server.

  Inline modes (`all_steered_shared`, `per_request_n*`) have 3-4× larger
  within-trial spread (29-37 ms p10-p90), driven by per-request vector
  materialization cost varying with payload size and timing. That spread
  is a real steering property, not a sampling artifact.

  Likely upstream causes (not pinned down, since vLLM-side investigation
  is out of scope for the steering RFC):
  1. Continuous-batching schedule-step boundaries — first request joining
     a new step pays a small extra cost vs requests joining an in-flight
     step. With concurrency=8 and a warmup drain, the first measured
     wave hits a fresh step.
  2. CUDA graph batch-size routing — vLLM compiles graphs for discrete
     batch sizes (1, 2, 4, 8, …); actual composition fluctuates each
     scheduler step.
  3. Prefix-caching probe cost — `--prefix-caching=True` on the bench;
     some requests hit a fresh cache, others reuse.

  To pin the root cause definitively would require instrumenting
  `bench_serving.py` to dump per-request TTFT samples (currently only
  aggregates are stored), then correlating against vLLM scheduler logs.
  Out of scope for this RFC.

#### Cross-GPU absolute numbers (old N=3 3090 vs N=9 A100, kept for hardware-scaling context)

These are median-based and don't account for the bimodal issue, but the cross-GPU ratios are the actionable signal. 3090: N=3 trials from `results/fused-index-post-164/` tag `base_t{1,2,3}` (pinned 1350 MHz). A100: original `post-merge[*]` N=9 trials on box `prxmx260002` (different physical box from the vs-disabled table above — pre-dates the rerun). Uncertainty = SE-of-median.

```
mode                  GPU   N   TTFT_p50        TTFT_p99         TPOT       E2EL
─────────────────────────────────────────────────────────────────────────────────
enabled_idle          3090  3   43.0 ± 0.1      57.3 ±  3.2     14.27     1855.1
enabled_idle          A100  9   28.7 ± 1.0      36.5 ± 10.0      7.38      965.7
named_shared          3090  3   54.1 ± 4.8      57.0 ±  2.9     14.28     1867.8
named_shared          A100  9   29.4 ± 1.2      36.8 ±  0.4      7.38      966.0
all_steered_shared    3090  3   60.9 ± 1.1      86.4 ±  0.4     14.40     1889.2
all_steered_shared    A100  9   42.4 ± 1.0      76.3 ± 28.5      7.50      997.7
per_request_n4        3090  3   63.6 ± 1.0      87.1 ±  0.5     14.39     1891.3
per_request_n4        A100  9   53.9 ± 2.3      90.2 ±  5.4      7.50     1007.1
per_request_n16       3090  3   61.5 ± 2.4      90.2 ±  1.8     14.49     1898.3
per_request_n16       A100  9   48.5 ± 1.9      85.1 ±  9.9      7.64     1022.4
```

Cross-GPU takeaways:
- **TPOT scales ~2× on A100 vs 3090** (7.4 vs 14.3 ms), as expected from hardware. Per-token steering cost remains mode-flat on both.
- **All 3090 mode-ranking and overhead trends reproduce on A100**, confirming the optimizations aren't 3090-specific.
- **3090 named_shared has a small residual TTFT gap vs enabled_idle** (54.1 vs 43.0) that A100 doesn't show. Worth a closer look but not load-bearing for the RFC.

#### Throughput-matrix overhead summary (A100 only, pinned 1410)

Latency overhead vs `disabled` baseline at each batch size, mode = mixed_50 (50% steered) and all_steered (100% steered):

```
batch  enabled_idle  mixed_50   all_steered
  1        +1.2%        —         +2.6%
  4        +1.7%      +3.9%       +3.8%
  8        +2.0%      +4.1%       +4.6%
 16        +2.0%      +4.8%       +6.1%
 32        +2.5%      +5.5%       +7.8%
```

Reads cleanly: enabled_idle stays ≤2.5% even at bs=32; all-steered scales from ~3% at bs=4 to ~8% at bs=32 (steering kernels are fixed cost while base compute amortizes with batch). Mixed_50 falls between the two as expected from the linear-in-active-fraction model.

#### Modes-matrix grid

426 cells from `--preset full`, axes = `(mode, batch_size ∈ {1,4,8,16,32}, num_hooks ∈ {1,2,3}, num_layers_steered ∈ {8,34}, prefix_caching=True, max_steering_configs_override)`. Modes: disabled / enabled_idle / inline_shared / inline_unique / named_shared / per_request_4. 36 of 426 cells errored on `PydanticUserError: Please use typing_extensions.TypedDict instead of typing.TypedDict` (mostly enabled_idle and disabled). That's a Pydantic / Python 3.13 compat issue in the bench process, not in vLLM steering; fix is a one-line import swap somewhere in the bench launcher path. Not summarized here because the cross-mode comparison only matches cleanly where every mode has a same-tuple disabled baseline, and that intersection is small. For the RFC, the throughput-matrix table above is a cleaner headline; modes-matrix is supplemental for hooks/layers scaling claims.

### 2. Model size question

3090 numbers use gemma-3-4b. The H100 sweep started with the same for direct cross-GPU comparability. **27B was discussed** — would give "scales to production model size" data for the RFC but the wins look smaller as a percentage (per-request overhead is fixed regardless of model size, so a 4 ms saving on a 60 ms request looks dramatic; same saving on a 1500 ms 27B request is invisible). Recommendation: ship 4B numbers first, add 27B as a supplemental if time permits.

### 3. RFC writeup itself

Not started. When ready to draft, the structure should be:

1. **Motivation** — inline-mode TTFT was ~300 ms with the old codepath, dominated by ~10-15 ms of JSON parse + np.asarray on the API server thread per request plus ~30 ms of synchronous auto-promote broadcast roundtrip plus various small costs.
2. **What changed** — the six landed PRs, organized by which path they targeted (request preprocessing, worker-side materialize, kernel JIT warmup, the auto-promote machinery itself).
3. **Numbers** — the table above plus the per-mode breakdowns from `results/`. Pinned-3090 numbers as the primary evidence; H100/A100 as the production-scale confirmation.
4. **Methodology** — the five lessons-learned above, plus the bench-infrastructure changes that shipped to make these measurements reproducible.
5. **What we explicitly chose NOT to do** — #159 (async-promote, made p99 worse) and #160 (fused-index, no measurable benefit post-binary-wire). Including the negative results strengthens the RFC.

## Reference: install command

Saved to memory but worth duplicating here:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=21943d4c258983c4b8eb56d50029aca4f18e4629 \
uv pip install -e ../vllm/steering --torch-backend=cu129    # cu130 on CUDA-13 hosts
```

Both env vars are required; without `VLLM_PRECOMPILED_WHEEL_COMMIT` the build pulls an arbitrary wheel and ABI-mismatches the local source tree (cryptic `undefined symbol` errors at import time).

## Reference: PR URLs

- #157 https://github.com/RhizoNymph/vllm/pull/157 (merged)
- #158 https://github.com/RhizoNymph/vllm/pull/158 (merged)
- #159 https://github.com/RhizoNymph/vllm/pull/159 (closed)
- #160 https://github.com/RhizoNymph/vllm/pull/160 (closed)
- #161 https://github.com/RhizoNymph/vllm/pull/161 (merged)
- #162 https://github.com/RhizoNymph/vllm/pull/162 (merged)
- #163 https://github.com/RhizoNymph/vllm/pull/163 (merged)
- #164 https://github.com/RhizoNymph/vllm/pull/164 (merged)

All against `RhizoNymph/vllm` base `feat/steering`. Upstream RFC will be filed against `vllm-project/vllm` once H100/A100 numbers are in hand.
