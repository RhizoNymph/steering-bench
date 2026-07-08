# Dynamic-steering overhead benchmark + the bs>16 cudagraph spike

Status: **in progress** (investigation running overnight 2026-06-17→18).
Model: `gemma-4-31B-it-Q4_K_S.gguf` (gemma4, hidden 5376, tp=1) on RTX 3090.
Bench: `scripts/bench_dynamic_steering.py` + `scripts/bench_static_steering.py`.

## FIX: fused non-mutating monitor (2026-06-19)

Implemented on `feat/dynamic-steering` (`1021417bba`): the in-graph monitor
gate is **fused into `apply_steering`** (gate computed in-kernel from the
pre-steering residual, folded into tier/row terms locally, never written
back) so the op is **non-mutating**; the separate mutating `steering_monitor`
is no longer emitted on the hot path. Also `bff44260ee`: made
`direct_register_custom_op` **idempotent** (cold torch.compile was
double-registering the GGUF op → hard crash; this blocked every cold-compiled
GGUF run). 101 CPU steering tests pass incl. 3 new fused-gate tests.

Live results (node0):
- **gemma-4-12B-it-GGUF (loads fine):** fused steer **+0.3% (bs24) / +0.7%
  (bs32)** == mutating **+1.5% / +0.7%**. The fix is **correct and
  non-regressing**, but the 12B *does not reproduce the spike* (~48 layers /
  faster steps → the steering ops don't trigger the FULL-graph downgrade the
  60-layer 31B does). So the 12B can't demonstrate the recovery.
- **gemma-4-31B (where the spike lives):** the recovery is pinned by the
  **monitor-drop diagnostic (+6.7%→+1.8% at bs32)** — the fused op's
  no-active-monitor path is that exact code path (no separate monitor op;
  fused gate skipped when inactive). A *direct* 31B fused run is blocked: the
  31B-Q4 peaks at the 3090's memory edge during GGUF weight-padding and now
  cold-loads OOM (the warm compile cache that had let it fit by a hair was
  cleared and can't be regenerated without a successful cold load). Not a
  steering issue.

Net: the fused op recovers the ~5% on the affected (large) model and doesn't
regress smaller ones. Cross-layer monitor (gate at layers > L) still uses the
mutating path and is the next step.

## TL;DR

- **True dynamic-steering overhead is <1% in steady state.** bs=1 shows
  +2.6–4.0% (fixed per-step consumer cost on a tiny ~2 s step), falling to
  +0.2–0.8% at bs=8 and bs=16. Capture-only stays ≤1% everywhere.
- A **+7–11% spike appears only with cudagraphs at decode batch >16**
  (bs=24/32/40). It is **not** steering compute: under `enforce_eager` the
  same bs=32 steering is **+0.5%**, and nsys shows **GPU kernel time is
  identical** to baseline — the cost is 100% host-side.
- **Introduced by dynamic steering, not pre-existing.** feat/integration's
  4-arg `apply_steering` is flat under cudagraphs (+0.4% at bs=32);
  dynamic-steering's 8-arg op spikes to +6.7% (static) / +8.6% (tier).
- **Root cause:** steering ops in the compiled layer forward downgrade the
  decode off the FULL cudagraph → ~2× more eager kernel launches per step
  (`cudaLaunchKernel` 6.7k→14.4k, `cudaGraphLaunch` 603→468). Host-launch
  overhead, hidden in eager / at bs≤16, exposed by fast graph replay at
  bs>16. **Fixable** (make the op cudagraph-fusable) → should reach the
  eager <1%. See §6.
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

feat/integration `steer` = static table gather; dynamic-steering
`steer_async` = the §5.4 dynamic tier (per-step `token_scales`/`dvec`).
Cudagraph, locked 1710, off vs steer:

| batch | feat/integration static | dynamic-steering tier (`steer_async`) |
|------:|----:|----:|
| 16 | +0.2% | +0.5% |
| 24 | +2.5% | **+10.8%** |
| 32 | **+0.4%** | **+8.6%** |

**feat/integration static steering is flat** — +0.2 / +2.5 / +0.4% across
bs=16/24/32 (the bs=24 +2.5% is a minor blip, not the spike). The
dynamic-steering tier is +8–11% at bs>16. **So the spike is NOT pre-existing
— dynamic steering introduced it.** (Same machine, same held 1710 MHz, same
model, same cudagraph mode; only the branch + steering path differ.)

Remaining question — *which* dynamic-steering change:
- the 8-arg `apply_steering` op (extra `scales`/`dvec`/`token_scales`/
  `row_gate` gather in the captured graph), or
- the dynamic-**tier** per-step machinery (runner writes `token_scales`/
  `dvec` each step; `steer_async` uses the tier, feat/integration has none).

Three-way comparison, cudagraph, locked 1710, off vs steer:

| batch | feat/int static (4-arg) | ds static (8-arg) | ds tier (`steer_async`) |
|------:|----:|----:|----:|
| 16 | +0.2% | +0.3% | +0.5% |
| 24 | +2.5% | +3.8% | +10.8% |
| 32 | **+0.4%** | **+6.7%** | **+8.6%** |

(bs=24 is a noisy/bad-bucket size across all configs — feat/int static is
+2.5% there but +0.4% at 32. Read bs=32 as the clean signal.)

### Verdict

**Introduced by dynamic steering.** feat/integration's lean 4-arg
`apply_steering` (`out = hidden + table[index]`) is essentially free under
cudagraph (+0.4% at bs=32). Dynamic steering's **8-arg op** —
`out = hidden + table[rows]*scale[rows]*row_gate + dvec*token_scales`, i.e.
**4 extra per-token buffer reads** (`steering_scales`, `steering_row_gate`,
`steering_dynamic_vec`, `steering_token_scales`) at every steered layer —
spikes to +6.7% at bs=32 *even for static steering with the tier at
defaults*. The dynamic tier's per-step `token_scales`/`dvec` writes add a
further ~2% (+8.6%). All of it is cudagraph-specific (eager ds-steer is
+0.5%) and only above bs=16.

So the regression is the **8-arg op's extra buffer reads in the captured
graph**, exposed once the GPU replay is fast enough (cudagraph) and the
batch is large enough (>16) that this per-token memory traffic / launch
shape stops being hidden. nsys (§6) pinpoints the exact mechanism.

Likely fix directions (to confirm with nsys): collapse the four extra
per-token gathers into fewer reads / fold defaults so the common path
(scale=1, row_gate=1, token_scales=0, dvec=0) reads less; or specialize the
kernel so an all-defaults invocation degenerates to the lean
`hidden + table[index]` path.

## 6. nsys root cause

nsys traces at **bs=32 cudagraph**, capture-range bracketed around the
measured iters (`scripts/nsys_steering_cell.py`, `--arm off|static|dynamic`),
`--trace=cuda,nvtx`. Reports: `cuda_gpu_kern_sum`, `cuda_api_sum`.

**GPU compute is identical across arms** — the dominant GGUF matmul
(`mul_mat_q4_K`) totals ~15.7–15.85 s in all three (off 15.71 / static 15.77
/ dynamic 15.85 s). So the +6.7–8.6% wall-clock is **entirely host-side
serialization**, not GPU work. Host-side `cuda_api_sum` deltas vs `off`:

| API call | off | static | dynamic |
|---|---|---|---|
| `cudaStreamSynchronize` | ~absent | 7.5 s / 372 (med **3.7 µs**) | **74 s / 288 (med 254 ms)** |
| `cudaLaunchKernel` | 2.3 s / **6,681** | 5.97 s / **14,442** | 3.5 s / 10,620 |
| `cudaGraphLaunch` | 603 | 468 | 468 |

Two host-side mechanisms, both introduced by enabling steering:

1. **Steering ops break the FULL cudagraph.** Static steering (no consumer)
   more than **doubles `cudaLaunchKernel`** (6.7k→14.4k) and *reduces*
   `cudaGraphLaunch` (603→468): the `apply_steering` custom op is not fused
   into the full graph, so steered layers fall back to extra per-layer eager
   launches. Its `cudaStreamSynchronize` is median ~3.7 µs (no-ops) — the
   cost is launch/overlap, not blocking. This is the bulk of static's +6.7%.
2. **The capture consumer adds heavy host-blocking syncs.** The dynamic arm
   (tier via `bench_steer_async`, which carries a global capture spec) shows
   `cudaStreamSynchronize` **median 254 ms × 288** — the host blocks mid-step.
   This is on top of (1) and drives dynamic to +8.6%. (Note capture-only with
   steering *disabled* was +1% in the 6-arm bench, so it's the
   capture+steering *combination* that stalls — likely the residual capture /
   APC decode-signature path interacting with steering.)

**Why only bs>16:** GPU compute is unchanged, so the fixed host overhead
(extra launches + syncs) is hidden while the host can race ahead of a slow
enough GPU. Under cudagraphs the replay is fast, and at bs>16 the per-step
host work stops being overlapped → exposed. (Eager pays per-op launch cost
anyway and the long forward hides it → +0.5%.)

### CONFIRMED: the mutating `steering_monitor` op is the dominant culprit

1-line diagnostic — skip emitting `torch.ops.vllm.steering_monitor` in
`apply_layer_steering` (node0, locked 1710, static steering):

| bs | feat/int (4-arg) | ds static +monitor | ds static **−monitor** |
|---:|----:|----:|----:|
| 24 | +2.5% | +3.8% | **+2.1%** |
| 32 | +0.4% | +6.7% | **+1.8%** |

At bs=32 the **mutating monitor op accounts for ~5% of the 6.7%**. The
residual +1.8% is the 8-arg `apply_steering` (~+1.4% over feat/int's 4-arg,
non-mutating → much friendlier). Tier vs static adds the final ~1.9%
(+8.6% tier − 6.7% static). So, by contribution at bs=32:
**monitor op ≈ 5% · 8-arg apply_steering ≈ 1.4% · tier machinery ≈ 1.9%.**

`steering_monitor` is `mutates_args=["steering_token_scales",
"steering_row_gate"]` and is emitted **unconditionally at every hook×layer
when `enable_steering=True`** (for stable topology / runtime toggling) —
even though an in-graph monitor is an opt-in Phase-2 feature almost never
configured. So the common steering user pays ~5% for a no-op mutating op
that's cudagraph-hostile.

**`auto_functionalized_v2=True` is NOT a usable fix here.** Tested both ways
on node0 (pass via `compilation_config`, and flip the vLLM default in
`compilation.py:904`): both **crash with a GGUF op double-registration**
(`vllm::_fused_mul_mat_gguf` registered twice at torch_utils.py:928) — the V2
functionalization transform re-registers the GGUF custom op. (This is why
vLLM defaults V2 off; its own comment flags V2 incompatibilities with custom
passes.) So we can't lean on the flag — the fix must be at the op level.

**Fix (high-leverage, low-risk): stop emitting `steering_monitor` unless a
monitor is actually configured.** Most usage has no monitor → the op
shouldn't be in the graph. Keep stable topology by deciding emission from
the operator-declared monitor sites at startup (recapture only if a monitor
is first configured at runtime — rare). That alone removes ~5%. Then trim
the 8-arg `apply_steering` (fold defaults so the common scale=1/row_gate=1/
token_scales=0/dvec=0 path reads fewer buffers, approaching the lean 4-arg
cost) for the residual ~1.4%.

### Primary driver (superseded — see "CONFIRMED" above)

The **FULL→eager/PIECEWISE cudagraph downgrade** is the main wall-clock cost,
and it hits **both** static and dynamic: with steering on, `cudaGraphLaunch`
*drops* (603→468) while `cudaLaunchKernel` ~doubles (6.7k→14.4k) — the
decode that ran as one graph replay now launches ~60 steering ops/layer (and
their neighbours) eagerly each step. GPU compute is unchanged, so this pure
host-launch overhead is hidden in eager (long forward) and at bs≤16 (host
races ahead of a slow-enough GPU) but exposed under cudagraphs at bs>16.

`apply_layer_steering`/`maybe_capture_residual` are called inside the
gemma4 decoder-layer `forward` (gemma4.py:732–773), which is inside the
`@support_torch_compile` region, and the steering ops are **not** in
`splitting_ops` — so this is a capture/replay downgrade, not a declared
split. The dynamic arm's 254 ms `cudaStreamSynchronize` (×288) is likely
**mostly background capture-dispatch-thread waits** (overlapped, not added
wall-clock — capture-only-no-steer was +1%); the static arm (no consumer,
median-3.7 µs syncs) shows the launch overhead alone is already +6.7%.

### Still open (morning)

- Confirm FULL→PIECEWISE downgrade directly: nsys-ui timeline of
  `/tmp/nsys_dyn_bt.nsys-rep` (saved on node0) + the cudagraph dispatch
  decision in `gpu_model_runner` (does the presence of the steering custom
  op drop the step out of the FULL graph at bs>16?). `--cudabacktrace`/
  `--python-backtrace` did **not** emit call-stack tables in this
  nsys 2025.3 sqlite export — use nsys-ui or `--sample=process-tree` next.

### Fix directions

- **Make `apply_steering` cudagraph-fusable / not a graph break** — the
  highest-leverage fix; would remove the extra launches (static's +6.7%).
  Check how the custom op is registered vs the `splitting_ops` list.
- **Decouple the capture-consumer sync from the step thread** when steering
  is active (the 254 ms blocking syncs) — keep capture dispatch fully async.
- Both are host-side; neither changes GPU work, so a fix should bring the
  cudagraph numbers down to the eager <1%.

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
