# Script migration status

Tracks the migration of `scripts/*.py` onto the Phase A engine seam
(`steering_bench.engine`) and Phase B harness (`steering_bench.harness`):

- **model dims** — sourced from `harness.models.get_model_config` instead of a
  per-script `MODEL_CONFIGS` table.
- **no `sys.path`** — relies on the installed package.
- **`--engine`** — takes an engine flag from the registry.
- **`engine` block** — stamps results with the first-class `engine` identity
  via `write_result(engine=...)`.

Status legend: **migrated** (all applicable items applied), **vllm-guarded**
(migrated onto central dims + engine block, generation is fork-specific so
`--engine` defaults to and is guarded to `vllm`), **exception** (intentionally
self-contained, documented below), **n/a** (analysis/tooling/vllm-internal — no
model dims / no result writing).

> **End state (Phase 7, final).** Every `scripts/*.py` is now one of:
> migrated, a documented `--engine vllm`-guarded fork-specific bench, an explicit
> branch-portable exception, part of the `scripts/vllm_internal/` suite, or an
> analysis/tooling helper. No script outside `scripts/vllm_internal/` defines a
> local `MODEL_CONFIGS` or a `sys.path.insert` (the only remaining
> `sys.path.insert` is in the `nsys_steering_profile.sh` shell wrapper, out of
> scope). Repo-wide grep counts (excluding `scripts/vllm_internal/`):
> `sys.path.insert` in `.py` → **0**; `MODEL_CONFIGS = {` → **0**;
> `from vllm import`/`import vllm` → only the documented `--engine vllm`-guarded
> benches plus two branch-portable exceptions (`bench_static_steering.py`,
> `nsys_steering_cell.py`) and the tooling helper cases noted below.

**Phase 4 (capture seam):** the public-API capture benchmarks now run through the
`CaptureEngine` / `CaptureSink` seams (`engine/capture.py`, `engine/capture_sink.py`),
and the `capture` capability *gates* discovery (`discover(required=Capabilities(capture=True))`),
so `run capture` and the capture scripts select only capture-capable engines. The
manager-internal capture micros stay in `scripts/vllm_internal/` (fork-coupled, no
cross-engine analog).

**Phase 5 (serving seam):** online/HTTP serving is now its own `ServingEngine` ABC
(`engine/serving.py`) — separate from `SteeringEngine` because the streaming
transport + `start_server`/`stop_server` lifecycle cannot fit the synchronous
`generate()` contract. Payload packing, the named-module register endpoint, and the
timing dump are **owned by the adapter** (pure `pack_steering_vectors` /
`named_register_payload` / `steering_extra_body` / metric functions) so scripts
never hand-encode. `bench_serving.py` is migrated onto it; the `serving` capability
was added to `Capabilities`, and `SteeringSpec` gained the Phase-3-deferred per-row
`scales` + a `PhaseSteeringSpec` companion for phase-split registration.

**Phase 6 (patch-sweep seam):** activation-patching / causal tracing is now a third,
distinct axis — a `PatchSweepEngine` ABC (`engine/patch_sweep.py`) separate from
`SteeringEngine` / `ServingEngine` because a denoising sweep returns a whole
`(layers × positions)` grid reduced to `cells / wall_s / cells_per_s` + an argmax
cell, not a latency profile. A frozen `PatchSweepResult` is the typed UNION of both
backends' fields (`noise_floor` / `auto_captured` / `skipped` are vLLM-only, default
`None` for TransformerLens); pure `from_tl_dict` / `from_vllm_dict` mappers + `to_dict`
reproduce each backend's original dict exactly. Two adapters wrap the verified
`external/` modules without rewriting them (`TLPatchSweepEngine` naive+batched,
`VllmPatchSweepEngine` one-call HTTP); `discover_patch_sweep` skips `tl` when
`transformer_lens` is absent and always lists `vllm` (server reachability checked at
`setup`). `Capabilities.patch_sweep` was added; `PatchSweepBenchmark`
(`harness/benchmarks/patch_sweep.py`, `PATCH_SWEEP_REGISTRY`) drives the comparison;
CLI `python -m steering_bench run patch-sweep`. `bench_patching_external.py` is migrated
onto it as a thin driver.

**Phase 7 (final — long-tail sweep):** now that every seam capability exists, the
remaining perf/profiling/correctness scripts were swept onto the same pattern:
drop `sys.path.insert`, source dims from `harness.models.get_model_config`
(dict-access → dataclass-attr), add an `--engine` flag that defaults to and is
guarded to `vllm` (these exercise fork-specific steering internals with no
cross-engine analog — the engine-agnostic successors already live in the
harness: `run latency|throughput|modes|capture|serving|patch-sweep`), and stamp
the `engine` block via `write_result(engine=VllmSteeringEngine().identity())`
where the script writes results. All known models each script referenced were
already present in the central `harness.models.MODEL_CONFIGS` table, so **no
additions** to the central table were required. `verify_correctness.py`,
`profile_steering.py`, and `nsys_target.py` write no result JSON, so they get the
dims + `sys.path` + `--engine` cleanup but no `engine` block.

## Headline scripts (Phase C target)

| Script | Status | Notes |
|--------|--------|-------|
| `bench_latency.py` | migrated | Dims from `harness.models`; no `sys.path`; `--engine` guarded to `vllm`; `engine` block stamped. As of **Phase 3** the engine-agnostic offline modes (`disabled`/`enabled_idle`/`inline_shared`/`inline_unique`/`named_shared`/`per_request_N`) are first-class in the seam+harness (`run latency --mode ...`); named modules + load-time steering config now live in the seam. Script retained for fork-specific nuances (prefix-cache isolation, auto-promote, mode×batch matrix). |
| `bench_throughput.py` | migrated | Same as `bench_latency.py`; engine-agnostic successor is `run throughput --mode ...` (Phase 3). Script retained for fork nuances. |
| `bench_memory.py` | migrated | Dims from `harness.models`; no `sys.path`; `--engine` guarded to `vllm` (measures fork steering buffers via `num_gpu_blocks_override` — no generation, no seam equivalent); `engine` block stamped. |
| `bench_external.py` | replaced (Phase 2) | Now a thin **deprecation shim** forwarding old flags to `python -m steering_bench run external-comparison`. The per-library `external/` adapters (hf_baseline, transformerlens, nnsight, repeng, pyvene, vllm_single/batched) and the `external/base.py::SteeringBenchmark` protocol were retired; each library is now a `SteeringEngine` adapter under `engine/engines/`. `external-comparison` covers both tiers via `--batch-size`. |

## Long-tail benches (Phase 7 — final)

Swept onto the central dims + `--engine`-guarded pattern. Each drops
`sys.path.insert`, resolves dims via `get_model_config(...).hidden_size /
.num_layers` (was `MODEL_CONFIGS[...][key]`), and takes `--engine` defaulted to
and guarded to `vllm` (these measure fork-specific steering internals; the
guard errors clearly on any other engine and points at the engine-agnostic
harness successor). Result-writing scripts stamp the `engine` block via
`VllmSteeringEngine().identity()`.

| Script | Status | Notes |
|--------|--------|-------|
| `bench_steering_modes_matrix.py` | vllm-guarded | Modes × workload cross-product; reuses `bench_throughput.run_throughput` per cell (resolved from the script dir on `sys.path[0]`). `engine` block stamped. Engine-agnostic successor: `run throughput --mode ...`. |
| `bench_throughput_matrix.py` | vllm-guarded | Mode × batch throughput matrix (mixed-fraction batches). `engine` block stamped. Successor: `run throughput`. |
| `bench_config_scaling.py` | vllm-guarded | `max_steering_configs` table-capacity sweep. `engine` block stamped. |
| `bench_cuda_graphs.py` | vllm-guarded | Steering op × CUDA-graph interaction (2×2 matrix). `engine` block stamped. |
| `bench_hook_points.py` | vllm-guarded | Active hook-point count sweep. `engine` block stamped. |
| `bench_max_tokens.py` | vllm-guarded | Per-step overhead vs `max_tokens`. `engine` block stamped. |
| `bench_mixed_batch.py` | vllm-guarded | Transitive-vs-proportional mixed-batch steering cost. `engine` block stamped. |
| `bench_table_sizing.py` | vllm-guarded | `max_cfg × batch × distinct` table sizing matrix. `engine` block stamped. |
| `bench_steering_with_capture.py` | vllm-guarded | Steering × capture matrix (Phase 4 seam-migrated). Phase 7 finished the tail: dropped `sys.path` + local `MODEL_CONFIGS`, added the `--engine vllm` guard, stamped the `engine` block. Subprocess-per-cell orchestration retained. Successor: `run capture`. |
| `bench_dynamic_steering.py` | vllm-guarded | Dynamic-steering/capture-consumer tiers (Phase 4 seam-migrated). Phase 7: dropped `sys.path`, added the `--engine vllm` guard, stamped the `engine` block (adapter identity + the script's own `_detect_vllm_commit`). Fork-only load knobs (`max_dynamic_steering_configs`, `enable_row_monitor`, action queue) stay vLLM-specific by design. |
| `profile_steering.py` | vllm-guarded | torch.profiler harness (writes Chrome traces, no result JSON → no `engine` block). Dims + `sys.path` + `--engine` cleanup applied. |
| `nsys_target.py` | vllm-guarded | Minimal nsys generate() target (no result JSON → no `engine` block). Dims + `sys.path` + `--engine` cleanup applied. |
| `verify_correctness.py` | vllm-guarded | Steering correctness checks (no result JSON → no `engine` block). Dims via `get_model_config` with a `ModelConfigError` guard replacing the old unknown-model branch; `--engine vllm` guard added. |

## Other axes (migrated in earlier phases)

| Script | Status | Notes |
|--------|--------|-------|
| `bench_serving.py` | migrated (Phase 5) | Online HTTP serving over the `ServingEngine` seam; `VllmServingEngine` owns subprocess launch, the async streaming driver, register/timing endpoints, and payload packing. `ServingBenchmark` drives it; `engine` block stamped. CLI: `run serving --engine vllm`. |
| `bench_patching_external.py` | migrated (Phase 6) | Cross-library causal-tracing sweep over the `PatchSweepEngine` seam (`tl_naive`/`tl_batched`/`vllm_sweep`); `PatchSweepBenchmark` owns the sweep + per-engine `write_result(engine=...)`. CLI: `run patch-sweep`. |
| `bench_capture_e2e.py` | migrated (Phase 4) | Public-API capture-overhead sweep via `configure_capture([CaptureConsumerSpec])` + per-request `RequestCapture`. Phase 7 removed a leftover `sys.path.insert`. Successor: `run capture`. |
| `bench_capture_filesystem.py` | migrated (Phase 4) | ActivationWriter throughput via the engine-neutral `CaptureSink` seam. Phase 7 removed a leftover `sys.path.insert`. |
| `bench_external.py` | replaced (Phase 2) | Thin deprecation shim forwarding to `run external-comparison`. |

## Exceptions and tooling (documented — no migration)

| Script | Status | Notes |
|--------|--------|-------|
| `bench_static_steering.py` | exception | **Branch-portable reproducer** — deliberately imports NOTHING from `steering_bench` (only `vllm` + numpy + stdlib) so the same file runs unchanged across fork branches; auto-detects hidden size from the loaded model (no `MODEL_CONFIGS`). The raw `from vllm import` and absence of `--engine` are by design; left untouched. |
| `nsys_steering_cell.py` | exception | Branch-portable single-config nsys target (only `vllm` + `SamplingParams.steering_vectors`, no dynamic-steering APIs). Raw `from vllm import` by design; no model dims, no result writing. In the boundary keep-list. |
| `bench_capture_serving.py` | n/a | Thin async HTTP client (`aiohttp`) driving a *running* server's capture path — no model construction, no model dims, no fork import, no `sys.path`. Nothing to migrate. |
| `analyze.py`, `analyze_kernel_isolation.py`, `compare_throughput.py`, `capture_throughput_calc.py`, `migrate_throughput_keys.py`, `rescale_clocks.py` | n/a | Analysis / tooling / profiling helpers — no model dims, no result writing, or already consume the shared schema. Phase 7 removed the trivially-safe `sys.path.insert` from `analyze.py` and `compare_throughput.py`. |

## vLLM-internal suite (intentionally not migrated)

Carved into `scripts/vllm_internal/` (Phase 1). These exercise vLLM-fork
internals directly — raw steering primitives and the capture-manager hot path
(`vllm.v1.capture.*`) — with **no model load and no cross-engine analog**. They
are deliberately **outside** the `steering_bench.engine` / `harness` framework
and are NOT candidates for the Tier-3 migration sweep; the `--engine` / seam
columns do not apply. See `scripts/vllm_internal/README.md`.

| Script | Fork internal measured |
|--------|------------------------|
| `vllm_internal/bench_steering_op.py` | Steering op kernel latency (`torch.ops.vllm.apply_steering` / reference impl). |
| `vllm_internal/bench_steering_manager.py` | `SteeringManager` Python-side overhead. |
| `vllm_internal/bench_index_building.py` | `steering_index` CPU construction loop. |
| `vllm_internal/bench_hash.py` | `hash_steering_config` cost (no argparse; runs on import). |
| `vllm_internal/bench_capture_manager.py` | `CaptureManager` plan build / GPU gather / dispatch. |
| `vllm_internal/bench_capture_latency.py` | Capture delivery latency (microbench + e2e). |
| `vllm_internal/bench_capture_plugin_work.py` | Per-chunk plugin work budget (microbench + e2e). |
| `vllm_internal/bench_capture_packed.py` | `per_file` vs `packed` filesystem-consumer layout throughput. |

## Post-migration correctness review + validation notes

A read-only correctness review of the seam→fork translation layers plus a
partial GPU validation were run after the 7 phases merged.

**Validated on hardware (RTX 3090):** the full framework spine end-to-end via a
real HF-engine run — `run latency`/`run external-comparison --engine hf_baseline`
on `Qwen/Qwen3-0.6B`: CLI dispatch, `Benchmark.run`, the engine seam, result
schema with the `engine` identity block, and `aggregate` (`engine_name` column).
The comparison table also confirmed the framework **gracefully skips an
installed-but-unusable engine** (`vllm` shows `FAILED`, `hf_baseline` succeeds).

**Not yet validated (needs GPU + a *built* vLLM fork):** all vLLM-dependent
paths. On the dev box the fork is editable-installed but the compiled extension
(`vllm._C`) is absent, so `import vllm` fails — the named-module RPC, capture
consumers, live serving server, and vLLM patch-sweep have not executed
end-to-end. These remain the user's acceptance step.

**Encoders verified byte-for-byte** against the pre-refactor scripts (named
registration payload + `(name, scale)` ref, serving base64 packing + TTFT/TPOT/
ITL/E2EL, capture consumer/request specs, patch-sweep dict mapping, load-time
config, result schema) — no payload-corruption bug found.

**Fixed** (silent-drop → loud error): (1) serving named-module references with a
non-default `scale` now raise instead of dropping the scale; (2) an offline
inline `SteeringSpec` carrying per-row `scales` now raises instead of applying
vectors unscaled (only the serving packer honors per-row scales).

**End-to-end validation on GPU (node0) — two follow-up fixes:**
- **`memory_mb: 0.0` for the vLLM engine.** vLLM V1 runs the model in an
  EngineCore *subprocess*, so `torch.cuda.memory_allocated()` in the driver
  process (what the base `_gpu_memory_mb` reads) sees 0 — a meaningless number
  for this engine. `VllmSteeringEngine.memory_allocated_mb` now reports
  *device-wide* used memory (`total - free` via `torch.cuda.mem_get_info`,
  factored into the pure, testable `device_used_memory_mb` helper), the honest
  measure given the subprocess architecture; it falls back to `0.0` when CUDA is
  unavailable. The base helper is unchanged — other engines load in-process and
  their per-process reading is correct.
- **patch-sweep `--base-url` inconsistency.** The serving path auto-appends
  `/v1`, but patch-sweep required the caller to pass a `/v1`-suffixed URL. A new
  `normalize_base_url` (in `external/vllm_patch_sweep.py`) accepts a bare host
  (`http://host:port`) or an already-`/v1`-suffixed URL and derives the `/v1`
  endpoint itself; `server_healthy` and `run_patch_sweep` are now robust to both
  forms and the CLI default is bare (`http://localhost:8000`), consistent with
  serving. Back-compat preserved for URLs already containing `/v1`. The
  `run_patch_sweep` request *body* is unchanged — a separate investigation owns
  it and a body fix may follow.

**Open caveats to confirm during GPU acceptance:**
- The migrated `bench_dynamic_steering.py` / capture scripts now pass
  `SteeringConfig` defaults (`enable_prefix_caching=True`, `max_steering_configs=4`)
  where the pre-refactor scripts passed neither, leaving fork defaults. Confirm
  the fork's defaults match; if not, thread the intended values explicitly, since
  prefix caching materially moves per-request steering cost.
- The engine-agnostic `run latency`/`run throughput` benchmarks steer a single
  `(layer, hook)`, whereas the headline `bench_latency.py`/`bench_throughput.py`
  steer all layers / multiple hooks. Their overhead numbers are **not directly
  comparable**; don't aggregate the two families as one series.

## Full-fork e2e validation (RTX 3090, fork `feat/integration`, gemma-3-4b-it)

All axes confirmed end-to-end against the real dynamic-steering fork build:
- **latency** (`inline_shared`, `named_shared` → register-module RPC + `(name,scale)` ref), **capture** (real activations captured), **serving** (own-server lifecycle, base64 packing, `/v1/steering/modules/register`, TTFT/TPOT/ITL/E2EL) — all pass.
- **dynamic-steering** — validated after the `feat/integration` merge (pure-Python, no rebuild): the action queue installs, `arm_active.tier_active=true` via status RPC; `off`/`steer_sync`/`steer_dynamic` all run.
- **patch-sweep** — validated: `72 cells @ ~59 cells/s, argmax L4@4`. The prior 400 was **not** a client bug — the client's one-call auto-capture body is correct. It required the server to be launched with all three of `--enable-patching --capture-consumers patch_source --patch-source-cache-bytes <N>`; a missing source-store cache silently dropped captured rows. The client now surfaces the server's error body + a launch-flag hint (`PatchSweepServerError`) instead of an opaque `raise_for_status`. See `docs/features/patch_sweep.md`.
