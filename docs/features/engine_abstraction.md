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
     engine.teardown()                            # frees GPU via external.base.cleanup_gpu
```

`SteeringSpec` is built from the `{hook:{layer:[floats]}}` dict that
`vectors.random_steering_vectors` emits via `SteeringSpec.from_vector_dict`,
validated once, and normalized to immutable tuples. Each adapter translates it
into its native form inside `generate`:

- vLLM: `spec_to_native(spec)` -> `{hook:{layer:[floats]}}` passed as
  `SamplingParams(steering_vectors=...)`; `NamedModuleRef` ->
  `SamplingParams(steering_module_ref=name)`. These translation functions are
  pure and do **not** import vllm.
- TransformerLens: `_resolve_single(spec)` enforces single-hook/single-layer,
  `_hook_name(hook, layer)` maps to `blocks.{layer}.{suffix}`, and an additive
  forward hook adds the vector each forward pass.

## Related Files

| File | Role | Key exports |
|------|------|-------------|
| `src/steering_bench/engine/spec.py` | Typed domain model + validation | `SteeringSpec` (`from_vector_dict`, `single`, `hooks`, `layers`, `dim`, `is_multi_hook`, `is_multi_layer`, `is_single_hook_single_layer`, `to_vector_dict`), `NamedModuleRef`, `GenerationRequest`, `GenerationResult`, `Steering`, `SteeringSpecError` |
| `src/steering_bench/engine/base.py` | Engine ABC + capability descriptor | `Capabilities` (+`satisfies`), `SteeringEngine` (`load`/`generate`/`memory_allocated_mb`/`teardown`/`version`/`commit`), `EngineError` |
| `src/steering_bench/engine/registry.py` | Data-driven, capability-aware discovery | `EngineEntry`, `ENGINE_REGISTRY`, `discover`, `is_package_available` |
| `src/steering_bench/engine/engines/vllm.py` | vLLM adapter + pure translation | `VllmSteeringEngine`, `spec_to_native`, `named_ref_to_kwargs`, `steering_kwargs` |
| `src/steering_bench/engine/engines/transformerlens.py` | TransformerLens adapter | `TransformerLensSteeringEngine`, `_resolve_single`, `_hook_name` |
| `src/steering_bench/engine/__init__.py`, `.../engines/__init__.py` | Re-exports | (public surface above) |
| `src/steering_bench/external/base.py` | Reused GPU helpers | `gpu_memory_mb`, `cleanup_gpu` (imported, not duplicated) |
| `tests/test_engine_spec.py`, `test_engine_registry.py`, `test_vllm_translate.py` | Unit tests (no GPU/backends) | — |

## Invariants / Constraints

- **Invalid states unrepresentable.** A `SteeringSpec` always has >=1 hook,
  each hook has >=1 layer, and all vectors within a hook share the same non-zero
  length; violations raise `SteeringSpecError` (a `ValueError`). Specs are
  frozen and inner vectors are normalized to `tuple[float, ...]`.
- **`GenerationRequest.max_tokens > 0`**; `GenerationResult.output_tokens >= 0`.
- **`Capabilities.satisfies(required)`**: an engine satisfies a requirement iff
  it provides every capability the requirement demands; a required field of
  `False` means "don't care".
- **Lazy backends.** Adapter modules must never import their heavy backend
  (vllm, transformer_lens, torch) at module scope — only inside method bodies —
  so `discover` and `spec_to_native` work with nothing installed.
- **TransformerLens** supports only single-hook/single-layer inline specs and
  rejects `NamedModuleRef` (`EngineError`); it generates sequentially.
- **vLLM load defaults** match the legacy adapters: `enable_steering=True`,
  `max_steering_configs=4`, `gpu_memory_utilization=0.9`, `max_model_len=2048`
  (overridable via `load(**opts)`).
- **Additive.** No existing module is modified except the one-line
  `output.py` except-clause bug fix; the `external/` package and
  `bench_external.py` remain the live path until Phase B/C migrate them.
