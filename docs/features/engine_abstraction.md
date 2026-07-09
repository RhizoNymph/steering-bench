# Engine Abstraction

A typed engine seam so the harness can benchmark inference engines beyond the
vLLM steering fork. A new engine is one adapter class; a benchmark can ask the
registry for "engines that support batching / capture / multi-hook…" and get
back only the ones that qualify.

## Scope

### In Scope (Phase A)
- Canonical, validated domain model for steering interventions and requests
  (`SteeringSpec`, `NamedModuleRef`, `GenerationRequest`, `GenerationResult`).
- `SteeringEngine` ABC with a `Capabilities` descriptor and the lifecycle
  `load` -> `generate` -> `teardown` plus memory/identity introspection.
- Capability-aware engine `registry` that generalizes the legacy
  `bench_external.discover_libraries` pattern.
- Two adapters: `VllmSteeringEngine` (full capabilities) and
  `TransformerLensSteeringEngine` (single-hook/single-layer, sequential).
- Pure, vllm-free `spec_to_native` translation so it is unit-testable with no
  backend installed.
- Purely additive: existing `external/` adapters, `scripts/bench_external.py`,
  and all existing scripts/tests are untouched and still work.

### Not In Scope (Phase A)
- Migrating existing scripts/benchmarks onto this seam — that is **Phase B**
  (`harness/` package: shared model config, args, measure loop) and **Phase C**
  (script migration + generic `engine` result block). Nothing is migrated yet.
- New engines beyond vllm/transformerlens (e.g. an `hf` baseline adapter).
- Timing/measurement, result-writing, or CLI — the engine layer only turns
  requests into results.

## Data / Control Flow

```
caller (future harness / bench script)
  → engine.registry.discover(filter_names, required=Capabilities(...))
       for each ENGINE_REGISTRY entry:
         skip if not in filter_names
         skip if entry.capabilities.satisfies(required) is False   (prints reason)
         skip if is_package_available(entry.required_package) is False (prints reason)
         importlib.import_module(entry.module_path); getattr(class)   (lazy)
       → returns list[type[SteeringEngine]]

  → engine = SomeEngineCls()
     engine.load(model_id, **opts)                # imports backend lazily here
     requests = [GenerationRequest(prompt, max_tokens, steering=SteeringSpec|NamedModuleRef|None)]
     results  = engine.generate(requests)         # -> list[GenerationResult]
     engine.memory_allocated_mb()
     engine.teardown()                            # frees GPU via engine.base.cleanup_gpu
```

`SteeringSpec` is built from the `{hook:{layer:[floats]}}` dict that
`vectors.random_steering_vectors` emits via `SteeringSpec.from_vector_dict`,
validated once, and normalized to immutable tuples. Each adapter translates it
into its native form inside `generate`:

- vLLM: `spec_to_native(spec)` -> `{hook:{layer:[floats]}}` passed as
  `SamplingParams(steering_vectors=...)`; `NamedModuleRef` ->
  `SamplingParams(steering_module_ref=(name, scale))` — the **`(name, scale)`
  tuple** the fork expects (Phase 3 fixed the earlier bare-name bug).
  `named_payload_from_spec(spec, prefill=, decode=)` builds the
  `{"vectors": {...}}` `register_steering_modules` payload (numpy-coercing).
  These translation functions are pure and do **not** import vllm.
- TransformerLens: `_resolve_single(spec)` enforces single-hook/single-layer,
  `_hook_name(hook, layer)` maps to `blocks.{layer}.{suffix}`, and an additive
  forward hook adds the vector each forward pass.
- HF baseline (`hf`): the no-op floor — real padded batching, steering accepted
  and **ignored**. Token counts are exact.
- nnsight: `resolve_single(spec)` + `layer_path(layer)` → residual add at
  `model.layers.{layer}.output[0]`. Single request → exact count; the pseudo-batch
  path (`batch_placeholder_results`) cannot recover per-prompt lengths, so it
  reports `max_tokens` and sets `output_tokens_exact=False`.
- repeng: `resolve_single` + `control_layers_for(layer, num_layers)` (a ~5-layer
  band) → `ControlModel` + `set_control`. The control object is built lazily and
  **cached keyed on the `SteeringSpec`** via `spec_cache_key(spec)` (a hashable
  content key); it is rebuilt only when the spec changes.
- pyvene: `resolve_single` + `component_for(hook)` (`pre_attn→block_input`,
  `post_attn`/`post_mlp→block_output`) → `IntervenableModel` + `AdditionIntervention`.
  The intervenable is likewise cached on the `SteeringSpec` (`spec_cache_key`) and
  rebuilt only on change.

`spec_to_native`, `resolve_single`, `spec_cache_key`, `control_layers_for`,
`component_for`, `layer_path`, and `batch_placeholder_results` are all pure,
heavy-lib-free module functions, so each adapter's translation / cache-key /
hook-mapping logic is unit-testable with nothing installed.

## Related Files

| File | Role | Key exports |
|------|------|-------------|
| `src/steering_bench/engine/spec.py` | Typed domain model + validation | `SteeringSpec` (`from_vector_dict`, `single`, `hooks`, `layers`, `dim`, `is_multi_hook`, `is_multi_layer`, `is_single_hook_single_layer`, `to_vector_dict`), `NamedModuleRef` (+`scale`), `GenerationRequest`, `GenerationResult` (+`output_tokens_exact`), `Steering`, `SteeringSpecError` |
| `src/steering_bench/engine/base.py` | Engine ABC + capability descriptor + load config + GPU helpers | `Capabilities` (+`satisfies`, +`prefix_cache`/`config_capacity`), `SteeringConfig`, `SteeringEngine` (`load(steering_config=)`, `register_module`), `EngineError`, `gpu_memory_mb`, `cleanup_gpu`, `is_library_available` (single definition; formerly `external.base`) |
| `src/steering_bench/engine/registry.py` | Data-driven, capability-aware discovery | `EngineEntry`, `ENGINE_REGISTRY` (6 engines), `discover`, `is_package_available` |
| `src/steering_bench/engine/engines/vllm.py` | vLLM adapter + pure translation | `VllmSteeringEngine` (`load(steering_config=)`, `register_module`), `spec_to_native`, `named_ref_to_kwargs`, `named_payload_from_spec`, `steering_kwargs` |
| `src/steering_bench/engine/engines/transformerlens.py` | TransformerLens adapter | `TransformerLensSteeringEngine`, `_resolve_single`, `_hook_name` |
| `src/steering_bench/engine/engines/hf.py` | HF no-op baseline adapter | `HFSteeringEngine` |
| `src/steering_bench/engine/engines/nnsight.py` | nnsight adapter | `NnsightSteeringEngine`, `resolve_single`, `layer_path`, `batch_placeholder_results` |
| `src/steering_bench/engine/engines/repeng.py` | repeng adapter (spec-keyed control cache) | `RepengSteeringEngine`, `spec_cache_key`, `control_layers_for`, `resolve_single` |
| `src/steering_bench/engine/engines/pyvene.py` | pyvene adapter (spec-keyed intervenable cache) | `PyveneSteeringEngine`, `spec_cache_key`, `component_for`, `resolve_single` |
| `src/steering_bench/engine/__init__.py`, `.../engines/__init__.py` | Re-exports | (public surface above) |
| `tests/test_engine_spec.py`, `test_engine_registry.py`, `test_vllm_translate.py`, `test_engine_hf.py`, `test_engine_nnsight.py`, `test_engine_repeng.py`, `test_engine_pyvene.py`, `test_generation_result.py` | Unit tests (no GPU/backends) | — |

## Invariants / Constraints

- **Invalid states unrepresentable.** A `SteeringSpec` always has >=1 hook,
  each hook has >=1 layer, and all vectors within a hook share the same non-zero
  length; violations raise `SteeringSpecError` (a `ValueError`). Specs are
  frozen and inner vectors are normalized to `tuple[float, ...]`.
- **`GenerationRequest.max_tokens > 0`**; `GenerationResult.output_tokens >= 0`.
- **`GenerationResult.output_tokens_exact`** defaults `True` (honest per-request
  count). An engine whose batch path cannot recover per-prompt lengths sets it
  `False` (nnsight's pseudo-batch) rather than reporting `max_tokens` as if exact.
- **Load-time-steering caching (repeng/pyvene).** The control/intervention object
  is built lazily on first `generate` and cached keyed on the `SteeringSpec`
  content (`spec_cache_key`); it is rebuilt only when the spec differs. A
  shared-vector workload pays the build once; a changing-vector workload
  legitimately incurs and measures the re-parameterization cost.
- **`Capabilities.satisfies(required)`**: an engine satisfies a requirement iff
  it provides every capability the requirement demands; a required field of
  `False` means "don't care".
- **Lazy backends.** Adapter modules must never import their heavy backend
  (vllm, transformer_lens, torch) at module scope — only inside method bodies —
  so `discover` and `spec_to_native` work with nothing installed.
- **TransformerLens** supports only single-hook/single-layer inline specs and
  rejects `NamedModuleRef` (`EngineError`); it generates sequentially.
- **vLLM load defaults** match the legacy adapters: `enable_steering=True`,
  `max_steering_configs=4`, `enable_prefix_caching=True` (via the default
  `SteeringConfig`), plus `gpu_memory_utilization=0.9`, `max_model_len=2048`
  (overridable via `load(**opts)`).
- **Named-module reference encoding (Phase 3).** `NamedModuleRef` carries a
  `scale: float = 1.0` (frozen, finite-validated); vLLM encodes it as the
  `(name, scale)` tuple in `SamplingParams.steering_module_ref`. The prior
  bare-name encoding was a real bug and is fixed.
- **Typed load-time steering config (Phase 3).** `SteeringConfig`
  (`enable_steering`/`max_steering_configs`/`enable_prefix_caching`) is the
  engine-agnostic load surface. `load(model_id, *, steering_config=None, **opts)`;
  each engine translates the fields it supports and **no-ops the rest**. Non-vLLM
  engines ignore `max_steering_configs`/`enable_prefix_caching` (they advertise
  `config_capacity=False`/`prefix_cache=False`); only vLLM sets both `True`.
  Backward-compatible: `load` without `steering_config` still works.
- **Named-module registration (Phase 3).** `register_module(name, spec, *,
  replace=True, prefill=None, decode=None)` defaults to raising
  `EngineError("named modules unsupported by <engine>")`; only engines with
  `named_modules=True` override it. vLLM calls
  `collective_rpc("register_steering_modules", {modules:{name:{"vectors":…}}, replace})`.
- **Deferred `SteeringSpec` extensions.** The optional per-row `scales` and
  `prefill`/`decode` splits on `SteeringSpec` are **deferred to Phase 5**
  (serving): they are only needed there, and adding them now would complicate the
  spec's construction/validation invariants for no in-phase consumer. The vLLM
  named payload already threads `prefill`/`decode` **specs** through
  `named_payload_from_spec` / `register_module` so the wire format is ready.
- **Phase 2 retirement.** The old `external/base.py::SteeringBenchmark` protocol
  and its per-library adapters were deleted; every steering library is now a
  `SteeringEngine` adapter here, and `scripts/bench_external.py` is a shim to
  `run external-comparison`. Only the patch-sweep modules (`external/tl_patching.py`,
  `external/vllm_patch_sweep.py`) remain under `external/` (Phase 6).
