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

## Headline scripts (Phase C target)

| Script | Status | Notes |
|--------|--------|-------|
| `bench_latency.py` | migrated | Dims from `harness.models`; no `sys.path`; `--engine` guarded to `vllm` (fork-only modes: `named_shared`, `inline_unique`, prefix caching, per-request `max_steering_configs`); `engine` block stamped. Engine-agnostic subset: `run latency`. |
| `bench_throughput.py` | migrated | Same treatment as `bench_latency.py`; `--engine` guarded to `vllm`. |
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
| `bench_steering_with_capture.py` | not-yet | Steering × capture matrix in subprocesses; local `MODEL_CONFIGS`; capture not in seam. Public-`LLM(capture_consumers=...)`-driven → Phase 4. |
| `profile_steering.py` | not-yet | nsys/profiling harness; local `MODEL_CONFIGS`. |
| `nsys_target.py` | not-yet | nsys profiling target; local `MODEL_CONFIGS`. |
| `verify_correctness.py` | not-yet | Correctness check, not a perf bench; local `MODEL_CONFIGS`. |
| `bench_dynamic_steering.py` | not-yet | Dynamic-steering/capture-consumer tiers; fork + entry-point consumers. |
| `bench_patching_external.py` | not-yet | Cross-library causal-tracing sweep; separate adapter set. |
| `bench_static_steering.py` | not-yet | Fork static-steering path. |
| `bench_capture_e2e.py`, `bench_capture_filesystem.py`, `bench_capture_serving.py` | not-yet | Public-API / writer capture benchmarks; capture not yet modeled by the seam → Phase 4. |
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
