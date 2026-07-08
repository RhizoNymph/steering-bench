# External Library Comparison

## Scope

Compare vLLM steering against TransformerLens, nnsight, repeng, pyvene, and a
HuggingFace no-op baseline, on one shared steered workload.

### In Scope
- Tier 1: single-request latency comparison across all engines (`--batch-size 1`).
- Tier 2: batched throughput comparison (`--batch-size N`), adding
  `req_per_sec` / `avg_per_request_ms`.
- HuggingFace baseline (no steering) as the floor.

### Not In Scope
- Semantic quality of steering (only performance).
- Engines not registered in `engine/registry.py`.
- The activation-patching / causal-tracing axis (`external/tl_patching.py`,
  `external/vllm_patch_sweep.py`) — a separate axis, migrated in Phase 6.

## Phase 2: seam-native

This benchmark is now **fully seam-native**. Each library is a `SteeringEngine`
adapter under `src/steering_bench/engine/engines/`, discovered through the engine
registry. The old `external/base.py::SteeringBenchmark` protocol and its
per-library adapters (`hf_baseline.py`, `nnsight_bench.py`, `repeng_bench.py`,
`pyvene_bench.py`, `transformerlens_bench.py`, `vllm_single.py`,
`vllm_batched.py`) were **deleted**. `scripts/bench_external.py` is now a thin
deprecation shim that forwards its old flags to the CLI.

```
python -m steering_bench run external-comparison --batch-size 1    # Tier 1
python -m steering_bench run external-comparison --batch-size 16   # Tier 2
```

## Data / Control Flow

```
python -m steering_bench run external-comparison [--batch-size N]
  → cmd_run peeks the benchmark, registers ExternalComparisonBenchmark.add_args
  → is_comparison=True → _run_comparison:
      discover() → every installed engine adapter
      for each engine:
        ExternalComparisonBenchmark(engine, config, batch_size=N).run()
          build_requests() → workload.steered_requests(..., batch_size=N)
          Benchmark base: load → warmup → measure (GPU-synced) → teardown
          extra_results(): for N>1, req_per_sec + avg_per_request_ms
          write_result(engine=engine.identity())   # engine block = adapter name
      _print_comparison_table(records)              # side-by-side mean/p90/tok-sec/mem
```

Each engine owns its `SteeringSpec` → native translation and hook-name mapping
(see `engine_abstraction.md`). The workload builds one inline `SteeringSpec`
(single hook/layer) and replicates it across the batch, so shared-vector caching
in repeng/pyvene pays the build once.

## Engine adapters

| Engine (`name`) | Batching | Steering mechanism | Notes |
|-----------------|----------|--------------------|-------|
| `hf_baseline` | Real (padded) batch | None (floor) | No-op; steering accepted and ignored. |
| `transformerlens` | Sequential | Additive forward hook | Single hook/layer. |
| `nnsight` | Pseudo-batch | Deferred trace, residual add | Batch path marks `output_tokens_exact=False`. |
| `repeng` | Sequential | `ControlModel` + `set_control` over a ~5-layer band | Control object cached on `SteeringSpec`. |
| `pyvene` | Sequential | `IntervenableModel` + `AdditionIntervention` | Intervenable cached on `SteeringSpec`. |
| `vllm` | Continuous batching | `SamplingParams.steering_vectors` | The fork. |

## Files

| File | Purpose |
|------|---------|
| `src/steering_bench/harness/benchmarks/external_comparison.py` | The comparison benchmark (`--batch-size`, Tier-2 metrics). |
| `src/steering_bench/harness/benchmarks/workload.py` | `steered_requests(batch_size=...)` workload builder. |
| `src/steering_bench/engine/engines/hf.py` | HF baseline adapter. |
| `src/steering_bench/engine/engines/nnsight.py` | nnsight adapter (pseudo-batch honesty flag). |
| `src/steering_bench/engine/engines/repeng.py` | repeng adapter (spec-keyed control cache). |
| `src/steering_bench/engine/engines/pyvene.py` | pyvene adapter (spec-keyed intervenable cache). |
| `scripts/bench_external.py` | Deprecation shim → `run external-comparison`. |

## Invariants

- Engines not installed are skipped at discovery time with a printed reason.
- The steered workload is one single-hook/single-layer `SteeringSpec` replicated
  across the batch (identical vectors), so repeng/pyvene build their control
  object once per run.
- `--batch-size 1` produces no Tier-2 metrics; `N>1` adds `req_per_sec` and
  `avg_per_request_ms`.
- Full GPU cleanup between engines via `engine.teardown()`.
