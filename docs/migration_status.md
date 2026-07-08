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
| `bench_external.py` | partial | Dims from `harness.models`; no `sys.path`; per-result `engine` block = library name. Intentionally keeps the broader `external/` per-library adapter set instead of the 2-engine registry. Seam-native successor already exists: `run external-comparison`. Does not take `--engine` (it iterates `--libraries`). |

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
| `bench_capture_manager.py` | not-yet | Capture-pipeline manager overhead; local `MODEL_CONFIGS`; capture not in seam. |
| `bench_steering_with_capture.py` | not-yet | Steering × capture matrix in subprocesses; local `MODEL_CONFIGS`; capture not in seam. |
| `profile_steering.py` | not-yet | nsys/profiling harness; local `MODEL_CONFIGS`. |
| `nsys_target.py` | not-yet | nsys profiling target; local `MODEL_CONFIGS`. |
| `verify_correctness.py` | not-yet | Correctness check, not a perf bench; local `MODEL_CONFIGS`. |
| `bench_dynamic_steering.py` | not-yet | Dynamic-steering/capture-consumer tiers; fork + entry-point consumers. |
| `bench_patching_external.py` | not-yet | Cross-library causal-tracing sweep; separate adapter set. |
| `bench_static_steering.py` | not-yet | Fork static-steering path. |
| `bench_steering_op.py`, `bench_steering_manager.py`, `bench_index_building.py`, `bench_hash.py` | not-yet | Microbenchmarks of raw steering primitives; no model load, no seam surface. |
| `bench_capture_*.py` (e2e, filesystem, latency, packed, plugin_work, serving) | not-yet | Capture-pipeline microbenchmarks; capture not modeled by the seam. |
| `analyze.py`, `analyze_kernel_isolation.py`, `compare_throughput.py`, `capture_throughput_calc.py`, `migrate_throughput_keys.py`, `nsys_steering_cell.py`, `rescale_clocks.py` | n/a | Analysis / tooling / profiling helpers — no model dims, no result writing, or already consume the shared schema. |
