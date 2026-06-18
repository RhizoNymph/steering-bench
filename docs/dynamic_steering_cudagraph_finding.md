# Dynamic-steering overhead benchmark + the bs>16 cudagraph spike

Status: **in progress** (investigation running overnight 2026-06-17→18).
Model: `gemma-4-31B-it-Q4_K_S.gguf` (gemma4, hidden 5376, tp=1) on RTX 3090.
Bench: `scripts/bench_dynamic_steering.py` + `scripts/bench_static_steering.py`.

## TL;DR

- **True dynamic-steering overhead is <1% in steady state.** bs=1 shows
  +2.6–4.0% (fixed per-step consumer cost on a tiny ~2 s step), falling to
  +0.2–0.8% at bs=8 and bs=16. Capture-only stays ≤1% everywhere.
- A **+8–11% spike appears only with cudagraphs at decode batch >16**
  (bs=24/32/40). It is **not** steering compute: under `enforce_eager` the
  same bs=32 steering is **+0.5%**. It is a **cudagraph-path artifact**.
- Attribution (feat/integration vs dynamic-steering) and nsys root cause:
  **see the live sections below.**
- ⚠️ **Housekeeping:** both GPUs were clock-locked at 1700 MHz for clean
  measurement (`sudo nvidia-smi -lgc 1700`). Restore with
  `sudo nvidia-smi -rgc` on localhost and node0 when done.

## 1. Clean results (overhead vs `off` = no capture, no steering)

bs=1 unlocked (boost ~1950 MHz, short cells, no throttle); bs=8/16/32 at
**locked 1710 MHz** (node0, which holds the lock under load — localhost
could not, see §3). All cudagraphs on unless noted.

| batch | capture_async | capture_sync | steer_async | steer_sync | steer_dynamic |
|------:|----:|----:|----:|----:|----:|
| 1  | +1.9% | +3.1% | +2.6% | +3.7% | +4.0% |
| 8  | +0.1% | −0.0% | +0.6% | +0.3% | +0.3% |
| 16 | +0.1% | −0.1% | +0.5% | +0.2% | +0.2% |
| 32 | +1.0% | +0.9% | **+8.6%** | **+8.4%** | **+8.4%** |

Tiers rank async ≤ sync ≤ dynamic at bs=1; converge into the noise floor at
batch. The bs=32 steering rows are the cudagraph artifact (§4), **not** real
cost.

## 2. Arms (see `src/steering_bench/capture_consumers/bench_consumers.py`)

`off` (no capture/steer) · `capture_async` (on_capture reader) ·
`capture_sync` (on_step reader) · `steer_async` (action-queue tier) ·
`steer_sync` (on_step tier) · `steer_dynamic` (in-graph monitor gating the
tier). Steering arms keep a **global** decode tier active so every decode
token is steered every step (steady-state); per-step policy work is one
fixed probe so the comparison isolates the transport.

## 3. Methodology: the thermal confound (why clocks are locked)

The first (unlocked) bs=32 runs showed ~+8% on the *later* cells of the
sweep regardless of arm — the jump hit `capture_sync` (cell 3, no steering)
on localhost and `steer_async` (cell 4) on node0. That was **thermal
throttling**: each bs=32 cell is ~3.6 min of sustained 100% load, so after
~2–3 cells the 3090 throttles and every later cell is ~8% slower. The `off`
baseline runs first (cool), inflating everything after it.

Fix: lock clocks (`nvidia-smi -lgc 1700`). **node0 holds 1710 under load**
(better cooling, 77 °C; verified live 1710 MHz @ 391 W, no throttle flags).
**localhost cannot** (83 °C; clock floats 1605–1665 per cell under bs≥16
load) — so localhost is unusable for clean bs>16 numbers and its locked
cross-check was discarded. bs=1 under a max-only `-lgc` is also noisy (light
load doesn't hold the cap → clock floats per cell); bs=1 uses the unlocked
boost run instead.

Once clocks were pinned on node0, the bs=32 jump **stopped being positional**
and instead landed exactly at the `enable_steering` boundary — revealing a
*real* effect that the thermal noise had been masking. Locking clocks was
what separated the two.

## 4. The cudagraph artifact

At locked 1710 MHz on node0, **eager vs cudagraph, bs=32, off vs steer_async:**

| mode | off | steer_async | overhead |
|---|---|---|---|
| cudagraph | 24689 ms | 26810 ms | **+8.6%** |
| eager | 23200 ms | 23322 ms | **+0.5%** |

So steering's real compute is +0.5%; the +8.6% is cudagraph-specific. Two
corroborating facts: (a) `off` itself is *slower* under cudagraphs at bs=32
(24689 vs eager 23200 — a general cudagraph-at-large-batch regression on
this GGUF, unrelated to steering), and (b) the inflation hits only steering
arms — capture arms stay ~1% under cudagraphs too.

**Batch localization (node0, locked 1710, cudagraph, off vs steer_async):**

| batch | steer overhead |
|------:|----:|
| 16 | +0.5% (clean) |
| 24 | **+10.8%** |
| 32 | +8.6% |
| 40 | +7.6% |

Sharp onset between bs=16 and bs=24, then the *percentage* declines while
absolute extra time grows slowly (~1.9→2.4 s) — the profile of a roughly
per-step host-side stall exposed only when the GPU replay is fast enough
(cudagraph) that CPU-side per-step work dominates. In eager the long forward
hides it.

**Ruled out:** the CUDA steering op is cudagraph-clean (always launches,
device-side `if active==0`, no host `.item()` — the `.item()` in
`apply_steering` is CPU-path only); no eager fallback / recompiles in logs
(only an unrelated `_compute_slot_mapping_kernel` JIT, once per cell); the
monitor (`steer_async`/`steer_sync` have none yet spike); table
re-population (tables populate once on tier-set, then only tiny per-token
pinned non_blocking H2D copies).

## 5. Attribution: pre-existing vs introduced by dynamic steering

Method: `scripts/bench_static_steering.py` — a branch-portable probe driving
**static** per-request steering via `SamplingParams.steering_vectors`
(base phase, prefill+decode), no dynamic-steering APIs. Run on node0
(holds 1710) on each branch via a temporary `vllm-fork` checkout (the
branches have **no C++ diff**, so the compiled `.so` is portable).

### feat/integration (4-arg `apply_steering`, pre-dynamic)

_(filling in as the run completes — bs=16,24,32, off vs steer)_

| batch | steer overhead |
|------:|----:|
| 16 | +0.2% |
| 24 | _pending_ |
| 32 | _pending_ |

### Verdict

_pending — see above._

## 6. nsys root cause

_pending — profiling the spiking config (bs=24 cudagraph, off vs steer),
comparing `cuda_api_sum` / `cuda_gpu_kern_sum` for the per-step stall._

## 7. Reproduce

```bash
# full matrix (locked clocks recommended; bs=32 needs a box that holds the lock)
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/bench_dynamic_steering.py \
  --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf --layer 30 --batch-sizes 1,8,16,32

# eager vs cudagraph isolation
... --arms off,steer_async --batch-sizes 32 [--enforce-eager]

# branch-portable static-steering probe
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/bench_static_steering.py \
  --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf --layer 30 --hidden 5376 \
  --batch-sizes 16,24,32
```

Raw per-cell JSON: `results/dynamic_steering/` (tagged
`split_*` / `locked1700_*` / `diag_*`).
