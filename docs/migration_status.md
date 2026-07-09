# Script migration status

Tracks the migration of `scripts/*.py` onto the Phase A engine seam
(`steering_bench.engine`) and Phase B harness (`steering_bench.harness`):

- **model dims** — sourced from `harness.models.get_model_config` instead of a
  per-script `MODEL_CONFIGS` table.
- **no `sys.path`** — relies on the installed package.
- **`--engine`** — takes an engine flag from the registry.
- **`engine` block** — stamps results with the first-class `engine` identity
  via `write_result(engine=...)`.

Status legend: **migrated** (all four applied), **partial** (some applied),
**not-yet** (untouched by Phase C).

**Phase 4 (capture seam):** the public-API capture benchmarks now run through the
`CaptureEngine` / `CaptureSink` seams (`engine/capture.py`, `engine/capture_sink.py`),
and the `capture` capability *gates* discovery (`discover(required=Capabilities(capture=True))`),
so `run capture` and the capture scripts select only capture-capable engines. The
manager-internal capture micros stay in `scripts/vllm_internal/` (fork-coupled, no
cross-engine analog).

## Headline scripts (Phase C target)

| Script | Status | Notes |
|--------|--------|-------|
| `bench_latency.py` | migrated | Dims from `harness.models`; no `sys.path`; `--engine` guarded to `vllm`; `engine` block stamped. As of **Phase 3** the engine-agnostic offline modes (`disabled`/`enabled_idle`/`inline_shared`/`inline_unique`/`named_shared`/`per_request_N`) are first-class in the seam+harness (`run latency --mode ...`); named modules + load-time steering config now live in the seam. Script retained for fork-specific nuances (prefix-cache isolation, auto-promote, mode×batch matrix) pending Phase 7. |
| `bench_throughput.py` | migrated | Same as `bench_latency.py`; engine-agnostic successor is `run throughput --mode ...` (Phase 3). Script retained for fork nuances pending Phase 7. |
| `bench_memory.py` | migrated | Dims from `harness.models`; no `sys.path`; `--engine` guarded to `vllm` (measures fork steering buffers via `num_gpu_blocks_override` — no generation, no seam equivalent); `engine` block stamped. |
| `bench_external.py` | replaced (Phase 2) | Now a thin **deprecation shim** forwarding old flags to `python -m steering_bench run external-comparison`. The per-library `external/` adapters (hf_baseline, transformerlens, nnsight, repeng, pyvene, vllm_single/batched) and the `external/base.py::SteeringBenchmark` protocol were retired; each library is now a `SteeringEngine` adapter under `engine/engines/`. `external-comparison` covers both tiers via `--batch-size`. |

## Remaining scripts (documented follow-up)

Not converted in Phase C. Each still defines a local `MODEL_CONFIGS` and/or a
`sys.path.insert`. They were left intact to avoid shipping unverifiable changes
(they require a GPU + the vLLM fork to run end-to-end, so the dict→dataclass
access-site rewrites can't be exercised here). Converting them is mechanical
follow-up: swap `MODEL_CONFIGS.get(...)[key]` for `get_model_config(...).<attr>`,
drop the `sys.path` hack, and pass `engine=` to `write_result`.

| Script | Status | Reason not migrated |
|--------|--------|---------------------|
| `bench_steering_modes_matrix.py` | not-yet | Fork-specific modes matrix; local `MODEL_CONFIGS`; unverifiable without GPU. |
| `bench_throughput_matrix.py` | not-yet | Fork throughput matrix; local `MODEL_CONFIGS`. |
| `bench_config_scaling.py` | not-yet | Fork `max_steering_configs` scaling; local `MODEL_CONFIGS`. |
| `bench_cuda_graphs.py` | not-yet | Fork CUDA-graph interaction; local `MODEL_CONFIGS`. |
| `bench_hook_points.py` | not-yet | Fork hook-point sweep; local `MODEL_CONFIGS`. |
| `bench_max_tokens.py` | not-yet | Fork max-tokens sweep; local `MODEL_CONFIGS`. |
| `bench_mixed_batch.py` | not-yet | Fork mixed-batch modes; local `MODEL_CONFIGS`. |
| `bench_table_sizing.py` | not-yet | Fork steering-table sizing; local `MODEL_CONFIGS`. |
| `bench_serving.py` | not-yet | Online HTTP serving internals; local `MODEL_CONFIGS`; no seam model. |
| `bench_steering_with_capture.py` | migrated (Phase 4) | Steering × capture matrix in subprocesses. LLM construction, steering vectors, capture consumers, and the per-request capture opt-in now route through the CaptureEngine seam (`VllmSteeringEngine.configure_capture(CaptureConsumerSpec)` + `SteeringConfig` + `GenerationRequest(steering=SteeringSpec, capture=RequestCapture)`); no raw `from vllm import`. Subprocess-per-cell orchestration retained. |
| `profile_steering.py` | not-yet | nsys/profiling harness; local `MODEL_CONFIGS`. |
| `nsys_target.py` | not-yet | nsys profiling target; local `MODEL_CONFIGS`. |
| `verify_correctness.py` | not-yet | Correctness check, not a perf bench; local `MODEL_CONFIGS`. |
| `bench_dynamic_steering.py` | partial (Phase 4) | Dynamic-steering/capture-consumer tiers. Consumer declaration is the typed `CaptureConsumerSpec`; LLM construction + activation-status introspection route through the CaptureEngine seam (`configure_capture` / `capture_status` / `live_capture_consumers`) — no raw `from vllm import LLM` / `collective_rpc`. The subprocess-per-cell arm orchestration and fork-only load knobs (`max_dynamic_steering_configs`, `enable_row_monitor`, action queue) stay vLLM-specific by design (irreducible fork coupling; no cross-engine analog). |
| `bench_patching_external.py` | not-yet | Cross-library causal-tracing sweep; separate adapter set. |
| `bench_static_steering.py` | not-yet | Fork static-steering path. |
| `bench_capture_e2e.py` | migrated (Phase 4) | Public-API capture-overhead sweep. Now builds `VllmSteeringEngine` via `configure_capture([CaptureConsumerSpec])` + per-request `RequestCapture`; no raw `from vllm import`. Seam-native equivalent: `python -m steering_bench run capture --capture-config ...` (`harness/benchmarks/capture.py`). |
| `bench_capture_filesystem.py` | migrated (Phase 4) | ActivationWriter throughput. Now drives the engine-neutral `CaptureSink` seam (`make_capture_sink("vllm", SinkConfig)` wrapping `ActivationWriter`, `WriteChunk`/`WriteFinalize`, `SinkThroughput`); no raw fork import. |
| `bench_capture_serving.py` | not-yet | Online HTTP capture path; deferred to the ServingEngine seam (Phase 5). |
| `analyze.py`, `analyze_kernel_isolation.py`, `compare_throughput.py`, `capture_throughput_calc.py`, `migrate_throughput_keys.py`, `nsys_steering_cell.py`, `rescale_clocks.py` | n/a | Analysis / tooling / profiling helpers — no model dims, no result writing, or already consume the shared schema. |

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
