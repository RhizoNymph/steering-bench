# Steering Subsystem Optimization Priorities

Derived from four nsys traces in `results/3090-timing/nsys/`
(`enabled_idle`, `named_shared`, `all_steered_shared`, `per_request_n4`),
each captured against `google/gemma-3-4b-it` on a 3090, NUM_PROMPTS=8,
CONCURRENCY=4, MAX_TOKENS=64. Host-timer dumps for each run are at
`<mode>.server.log`.

## Cross-mode cost summary

Numbers are wall-time inside the captured request window (post-cuda_start
to cuda_stop). `n` is the count of NVTX firings.

| metric | enabled_idle | named_shared | all_steered_shared | per_request_n4 |
|---|---|---|---|---|
| `update_steering_buffers.total` | n=130 / 1.2 ms | n=129 / 42.5 ms | n=145 / 71.1 ms | n=142 / **81.5 ms** |
| `register_initial_steering_config` | n=8 / 0.2 ms | n=8 / 16.8 ms | n=8 / 18.3 ms | n=8 / 19.8 ms |
| `register_config.materialize` | — | n=5 / 17.4 ms | n=9 / 18.8 ms | **n=16 / 23.0 ms** |
| `stack_vectors_to_device` | — | n=5 / 16.2 ms | n=9 / 16.7 ms | n=16 / 19.0 ms |
| `populate.scatter` (incl. index_copy_loop) | — | n=5 / 14.6 ms | n=10 / 29.0 ms | n=11 / 32.1 ms |
| `populate.any_active_writes` | — | n=5 / 10.1 ms | n=10 / 19.7 ms | n=11 / 24.6 ms |
| `auto_promote.prep` | n=8 / 0.1 ms | n=8 / 0.1 ms | n=8 / **32.6 ms** | n=8 / **40.8 ms** |
| `auto_promote.register_rpc_async` | — | — | **n=1 / 44.0 ms** | **n=4 / 76.1 ms** (max 32.2) |
| `pack_inline_steering` | — | — | n=1 / 3.0 ms | n=4 / 11.8 ms |
| Triton JIT (`cuLibraryLoadData`) | 2 / 17.7 ms | 2 / 20.1 ms | 2 / 24.6 ms | 2 / 24.7 ms |
| GPU idle gaps >1ms | 5 / 51 ms / max 24 | 5 / 52 ms / max 28 | 5 / **102 ms** / max 51 | 8 / **102 ms** / max **69** |

Key reads:

- **`enabled_idle` is the floor.** All steering host work is short-circuited
  (~1.2 ms across the bench). Only cost left is the 18 ms of Triton JIT and
  the standard request-boundary RTT gaps. Demonstrates the no-active-config
  path is clean.
- **`named_shared` only pays the cold-path cost on first arrival.** 5
  materialize calls for 8 requests means refcount-hit dominates; 14
  named-cache-hit resolves vs 0 inline.
- **`all_steered_shared` converges to named_shared after 2 strikes.** Just
  2 inline resolves before auto-promote fires (`register_rpc_async` once,
  44 ms blocking), then 14 named-cache-hits. Single-shared-vector mode
  pays the promotion tax once and then runs at named_shared cadence.
- **`per_request_n4` pays everything per batch.** 4 distinct configs →
  4 promote RPCs (76 ms cumulative, max 32 ms each, **fully GPU-blocking**),
  16 materializes, 4 inline packs (12 ms), and 142 update_buffers (twice
  the populate.scatter count vs named_shared). The 69 ms GPU-idle gap at
  1462 ms maps directly to the first promote RPC firing.

## Scope

## Scope

What follows is the list of costs in the steering subsystem that are
**addressable and likely to move TTFT**. Costs that the trace shows are
out of the steering subsystem's control (client RTT, APIServer↔engine
IPC handoff, decode-step GPU compute, vLLM scheduler memcpy traffic) are
called out explicitly at the bottom so they don't get re-litigated.

The trace context: a captured window of ~1.6 s containing 8 served
requests, 129 forward passes, 231 cudagraph replays. The apply_steering
Triton kernel runs **inside the cudagraph replays** — it appears zero
times as a standalone `cuLaunchKernelEx` event during the window. The
steady-state per-decode-step steering cost is therefore essentially
free; everything below is **transition / cold-path** cost.

## Reordered priorities (after multi-mode capture)

The named_shared-only ranking in the original draft was wrong for the
modes that actually drive production TTFT regressions. Updated order:

## Priority 1 — Make `auto_promote.register_rpc_async` fire-and-forget

**Cost in traces:**
- `per_request_n4`: **76 ms / 4 calls, max 32 ms** — directly maps to the
  69 ms GPU idle gap at 1462 ms (the GPU has nothing to do because the
  engine thread is blocked on the broadcast).
- `all_steered_shared`: 44 ms / 1 call — 51 ms GPU idle gap at 1467 ms.
- `named_shared` / `enabled_idle`: 0 — never fires.

**Why #1:** in inline-vector modes this is the single biggest blocking
event in any trace, and it sits on the request critical path. The
broadcast is logically a side effect — the named name only starts being
referenced on *subsequent* requests, so blocking the request that
triggered the promotion is unnecessary.

**Action:** dispatch the worker-broadcast asynchronously
(`asyncio.create_task` or equivalent) and return from the API handler
immediately. Naming-conflict races between concurrent promoters of the
same hash are already handled by the manager's refcount; nothing in the
return path needs the broadcast to have completed.

**Where:**
`vllm/config/steering_types.py::maybe_auto_promote_steering_modules`
(both sync and async variants); the API router preprocessing call site.

## Priority 2 — Pre-materialize named modules at register time

**Cost in trace:** 15 ms first-time `manager.register_config.materialize`
on the critical path of request 1 in `named_shared`. Repeats for every
cold (hash, phase) pair that lands. Refcount-hit afterwards is
near-zero.

**Why #2:** biggest TTFT win for the named_shared production pattern
(stable named modules registered ahead of time but materialized
lazily). Fix is small and isolated.

**Trace evidence:** `stack_vectors_to_device` is 14.9 ms of the 15 ms —
single H2D of 34 layers × hidden_size of bf16. Same shape in every mode
(11–15 ms first call) regardless of whether vectors arrive inline or
via name.

**Action:** allocate the device row + upload vectors at
`/v1/steering/modules/register` time so the first request resolving to
the module finds a refcount-hit instead of a cold materialize.

**Where:** `vllm/v1/worker/steering_manager.py` register/materialize
path, called from
`vllm/v1/worker/steering_model_runner_mixin.py::_register_initial_steering_config`.
Trigger added to
`vllm/entrypoints/serve/steering/api_router.py::register_steering_modules`.

## Priority 3 — Pre-warm Triton JIT for all served shapes

**Cost across all 4 traces:** exactly 2 `cuLibraryLoadData` events per
mode, 17–25 ms cumulative. The count is **identical even in
`enabled_idle`** (no active steering vectors), confirming the JIT is
shape-driven, not data-driven.

| mode | JIT total | first @ | second @ |
|---|---|---|---|
| enabled_idle | 17.7 ms | 197 ms | 957 ms |
| named_shared | 20.1 ms | 218 ms | 1753 ms |
| all_steered_shared | 24.6 ms | 602 ms | 1376 ms |
| per_request_n4 | 24.7 ms | 603 ms | 1367 ms |

**Why it matters:** the `do_not_specialize=["N","H"]` change cut variant
count from 3→2 but did not eliminate it. The two JITs are not random —
they fire on first-prefill and again later (probably first-decode at a
new batch size), regardless of mode. Pre-touching both shapes during
warmup eliminates the 20 ms from every cold run, mode-agnostic.

**Action:**
1. Dump `_apply_steering_kernel.cache` after a representative bench
   run to identify which arg(s) still drive specialization between the
   two surviving variants.
2. Either expand `do_not_specialize` / `do_not_specialize_on_alignment`
   to cover the remaining arg, or extend
   `warmup_apply_steering_kernel` in `steering_kernel.py` to invoke the
   kernel at every shape vLLM will subsequently hit (drive it from the
   `cudagraph_capture_sizes` list so it stays in sync).

**Where:** `vllm/model_executor/layers/steering_kernel.py`.

## Priority 4 — Async `stack_vectors_to_device`

**Cost in traces:**
- `named_shared`: 16.2 ms / 5 calls (cold-path only).
- `all_steered_shared`: 16.7 ms / 9 calls.
- `per_request_n4`: **19.0 ms / 16 calls** — every distinct config that
  lands needs the H2D.

**Why:** the current path is essentially `torch.tensor(list,
device=cuda)` semantics — implicitly synchronous H2D. The H2D itself is
unavoidable (vectors do need to land on device), but **blocking the
caller** is not.

**Action:** pinned-CPU staging buffer + `non_blocking=True` copy +
defer the synchronization to populate time (which already has a sync
boundary). This lets the H2D overlap with whatever GPU work is in
flight from the previous step.

**Where:** `vllm/v1/worker/steering_manager.py::_stack_vectors_to_device`.

## Priority 5 — Fuse the per-layer `index_copy_` loop

**Cost in traces:**
- `named_shared`: 13.3 ms / 5 calls (~2.7 ms each).
- `all_steered_shared`: 26.5 ms / 10 calls.
- `per_request_n4`: **28.9 ms / 11 calls**, max 3.2 ms.

The per-call cost is *roughly constant across modes* (~2.6–2.9 ms). The
total grows with how often populate has to run, not with active-config
count — so this is launch-overhead bound, not data bound.

**Why:** with 34 layers × 1 hook × 1 active config, the scatter does 34
sequential `index_copy_` launches per call. At ~80 µs each
(launch + tiny copy) this is the dominant per-populate cost.

**Action:** stack per-layer tables into a single `[L, R, H]` tensor and
do one `index_copy_` along dim=1. Worth doing even at 1 active config —
the launch overhead is the cost.

**Where:** `vllm/v1/worker/steering_manager.py::populate_steering_tables`
(the homogeneous-dtype fast path added during the prior optimization
pass).

## Priority 6 — Reduce `pack_inline_steering` cost

**Cost in traces:**
- `enabled_idle` / `named_shared`: not present (no inline payload).
- `all_steered_shared`: 3.0 ms / 1 call (one pack before promotion).
- `per_request_n4`: 11.8 ms / 4 calls (~3 ms each).

**Why:** inline modes pay 3 ms per request to repackage the vector
payload into the canonical request format on the API server side. At
4 RPS this is 12 ms of APIServer-thread time per second — small but
nonzero.

**Action:** profile the pack to identify whether it's hashing,
JSON re-serialization, or NumPy conversion. Likely `np.asarray(...)` on
deeply-nested Python list-of-lists.

**Where:** `vllm/config/steering_types.py::maybe_pack_inline_steering_for_request`.

## Out of scope (NOT addressable in steering subsystem)

These show up in the trace but live elsewhere — listed so they don't
get re-investigated:

- **~9ms gap at 989→998ms.** Client RTT (httpx iterating SSE to EOF
  and POSTing the next request) plus APIServer→engine IPC handoff plus
  4× APIServer-side `frontend.auto_promote.prep` (~700µs total). Engine
  CUDA thread idle the whole time. Would happen with steering disabled.
- **~11.6ms decode cadence.** Pure forward-pass GPU compute on the
  3090. Apply_steering is captured into the cudagraph and contributes
  no measurable per-step cost during replay.
- **2,463 `cudaMemcpyAsync` calls (~34ms total).** vLLM scheduler
  per-step input metadata (sampling params, positions, slot mappings).
- **2.4s of `cudaEventSynchronize` aggregated across threads.**
  Engine bookkeeping, not steering.
- **Piecewise cudagraph splits around attention.** The
  "graph→kernel→graph" pattern is the documented behavior of
  `splitting_ops` (unified_attention_with_output etc.), not a steering
  break.

## Mode-specific applicability

| Priority | enabled_idle | named_shared | all_steered_shared | per_request_n4 |
|---|---|---|---|---|
| 1. async auto-promote RPC | — | — | ✓ (44 ms) | ✓✓ (76 ms) |
| 2. pre-materialize named | — | ✓✓ (15 ms) | ✓ (after promote) | — |
| 3. JIT pre-warm | ✓ (18 ms) | ✓ (20 ms) | ✓ (25 ms) | ✓ (25 ms) |
| 4. async H2D | — | ✓ | ✓ | ✓✓ (16 calls) |
| 5. fused index_copy | — | ✓ | ✓✓ | ✓✓ |
| 6. pack_inline cost | — | — | minor | ✓ (12 ms) |

## Suggested order of attack

1. **Priority 1 (async auto-promote RPC)** — biggest single
   inline-mode TTFT win; eliminates the 76 ms blocking event in
   `per_request_n4` and the 44 ms event in `all_steered_shared`. Small
   patch.
2. **Priority 3 (JIT pre-warm)** — flat 18–25 ms off *every* cold run
   regardless of mode, including `enabled_idle`. Easiest to verify
   (count drops to 0 in next trace).
3. **Priority 2 (pre-materialize named)** — biggest named_shared TTFT
   win; eliminates 15 ms cold-start cost on first request to each
   named module.
4. **Priority 5 (fused index_copy)** — recurring per-populate cost in
   active-steering modes; ~26–29 ms cumulative in inline modes,
   13 ms in named_shared. Cleanest microbench target.
5. **Priority 4 (async H2D)** — overlap rather than eliminate. Smaller
   absolute win than 1–3 but applies broadly.
6. **Priority 6 (pack_inline cost)** — only meaningful for inline modes
   at high RPS; investigate after the bigger fish are off the line.
