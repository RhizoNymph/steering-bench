# steering-bench

## Overview

### Description

A standalone benchmark harness that measures the performance overhead of vLLM's activation steering feature. Compares vLLM steering against external libraries (TransformerLens, nnsight, repeng, pyvene) and measures interactions with vLLM optimizations (CUDA graphs, torch.compile, prefix caching).

Completely external to the vLLM codebase — vLLM is an optional dependency.

### Subsystems

- **Core (`src/steering_bench/`)**: Shared utilities for timing, output, and vector generation. All benchmarks depend on these.
- **Benchmark Scripts (`scripts/`)**: Plain Python scripts that run individual benchmarks. Each is independently executable.
- **Results (`results/`)**: Gitignored directory where benchmark JSON output is written.
- **Docs (`docs/`)**: Design documents, roadmap, and feature documentation.

### Data Flow

```
[Benchmark Script] 
    → imports core utilities (timing, output, vectors)
    → loads vLLM model (or external library)
    → runs timed iterations
    → writes JSON result via output.write_result()
    → prints human-readable summary
```

All results share a common JSON schema with environment metadata, parameters, and timing statistics. Records also carry a first-class, cross-engine `engine` block (`{name, version, commit}`) written via `output.write_result(engine=...)`; `analysis/aggregate.py` surfaces it as `engine_name` / `engine_version` / `engine_commit` columns. The legacy `environment.vllm_version` / `environment.vllm_commit` fields remain for backward compatibility, but `engine` is the canonical identifier for grouping/comparing results across engines. Records predating the block load fine (columns are `None`).

## Features Index

### core
- description: Shared benchmark infrastructure — GPU-synced timing, JSON output schema, steering vector generation
- entry_points: [src/steering_bench/timing.py, src/steering_bench/output.py, src/steering_bench/vectors.py]
- depends_on: []
- doc: docs/features/core.md

### micro_benchmarks
- description: Microbenchmarks for raw steering primitives (op kernel, manager, index building)
- entry_points: [scripts/bench_steering_op.py, scripts/bench_steering_manager.py, scripts/bench_index_building.py]
- depends_on: [core]
- doc: docs/features/micro_benchmarks.md

### vllm_benchmarks
- description: End-to-end vLLM system benchmarks (latency, throughput, memory). As of Phase C the three headline scripts drop the `sys.path` hack, resolve model dims via `harness.models.get_model_config`, take `--engine` (guarded to `vllm` since the modes are fork-specific), and stamp results with the first-class `engine` block. The engine-agnostic subset runs via `python -m steering_bench run latency`.
- entry_points: [scripts/bench_latency.py, scripts/bench_throughput.py, scripts/bench_memory.py, scripts/bench_steering_with_capture.py]
- depends_on: [core, engine_abstraction, harness]
- doc: docs/features/vllm_benchmarks.md

### steering_modes_matrix
- description: Cross-product bench of steering modes (named_shared, inline_shared, inline_unique, per_request_4, enabled_idle, disabled) × {batch_size, num_hooks, layer_subset, prompt_len}.  One sweep produces the comparison data for "did the new optimizations close the gap?"
- entry_points: [scripts/bench_steering_modes_matrix.py]
- depends_on: [core, vllm_benchmarks]
- doc: docs/features/steering_modes_matrix.md

### ablation_benchmarks
- description: Optimization interaction tests (CUDA graphs, config scaling, hook points)
- entry_points: [scripts/bench_cuda_graphs.py, scripts/bench_config_scaling.py, scripts/bench_hook_points.py]
- depends_on: [core]
- doc: docs/features/ablation_benchmarks.md

### external_comparison
- description: Cross-library steering performance comparison (TransformerLens, nnsight, repeng, pyvene vs vLLM). `scripts/bench_external.py` keeps the broader per-library `external/` adapter set (dims now from `harness.models`, each result stamped with an `engine` block = library name). Its engine-seam-native successor is the `external-comparison` harness benchmark: `python -m steering_bench run external-comparison`.
- entry_points: [scripts/bench_external.py, src/steering_bench/harness/benchmarks/external_comparison.py]
- depends_on: [core, engine_abstraction, harness]
- doc: docs/features/external_comparison.md

### engine_abstraction
- description: Typed engine seam — a `SteeringEngine` ABC with a `Capabilities` descriptor, a canonical `SteeringSpec`/`NamedModuleRef` domain model, a capability-aware engine registry, and vLLM + TransformerLens adapters. Lets a new engine be one adapter class and lets the registry filter engines by required capabilities. Additive in Phase A; migrating existing scripts onto this seam is Phase B/C.
- entry_points: [src/steering_bench/engine/spec.py, src/steering_bench/engine/base.py, src/steering_bench/engine/registry.py, src/steering_bench/engine/engines/vllm.py, src/steering_bench/engine/engines/transformerlens.py]
- depends_on: [core]
- doc: docs/features/engine_abstraction.md

### harness
- description: Benchmark harness built on the engine seam — a single `get_model_config` (static table + HuggingFace AutoConfig fallback) for model dims, shared `add_common_args` argparse flags, a `Benchmark` base class owning the warmup->measure->write->print lifecycle, and a `steering-bench` CLI (`run <benchmark> --engine <engine>`, `list`, `engines`) with a benchmark registry. Two proof benchmarks (`latency`, `external-comparison`) run engine-agnostically. Additive; migrating the old `scripts/*.py` onto the harness is Phase C.
- entry_points: [src/steering_bench/harness/models.py, src/steering_bench/harness/args.py, src/steering_bench/harness/benchmark.py, src/steering_bench/harness/benchmarks/registry.py, src/steering_bench/__main__.py]
- depends_on: [core, engine_abstraction]
- doc: docs/features/harness.md

### analysis
- description: Result aggregation and matplotlib chart generation
- entry_points: [scripts/analyze.py]
- depends_on: [core]
- doc: docs/features/analysis.md

### capture_consumers
- description: Benchmarks for vLLM's activation capture pipeline — manager overhead (plan build, GPU gather, dispatch) and filesystem writer throughput
- entry_points: [scripts/bench_capture_e2e.py, scripts/bench_capture_manager.py, scripts/bench_capture_filesystem.py]
- depends_on: [core]
- doc: docs/features/capture_consumers.md

### dynamic_steering
- description: End-to-end overhead of the dynamic-steering / capture-consumer tiers vs a no-capture/no-steering baseline on a gemma4 model. Six arms (off, capture_async, capture_sync, steer_async, steer_sync, steer_dynamic) decompose capture-pipeline cost, sync-vs-async on_step cost, the steering kernel, and the three tier transports head-to-head. Doubles as a capture-consumer perf benchmark.
- entry_points: [scripts/bench_dynamic_steering.py, src/steering_bench/capture_consumers/bench_consumers.py]
- depends_on: [core, capture_consumers]
- doc: docs/features/dynamic_steering_bench.md

## Performance / Investigation Docs

- `docs/performance.md` — consolidated steady-state perf characteristics and prior optimization passes
- `docs/optimization_priorities.md` — addressable steering-subsystem costs surfaced by the 3090 nsys trace, ranked by TTFT impact
