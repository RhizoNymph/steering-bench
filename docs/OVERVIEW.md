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

All results share a common JSON schema with environment metadata, parameters, and timing statistics.

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
- description: End-to-end vLLM system benchmarks (latency, throughput, memory)
- entry_points: [scripts/bench_latency.py, scripts/bench_throughput.py, scripts/bench_memory.py]
- depends_on: [core]
- doc: docs/features/vllm_benchmarks.md

### ablation_benchmarks
- description: Optimization interaction tests (CUDA graphs, config scaling, hook points)
- entry_points: [scripts/bench_cuda_graphs.py, scripts/bench_config_scaling.py, scripts/bench_hook_points.py]
- depends_on: [core]
- doc: docs/features/ablation_benchmarks.md

### external_comparison
- description: Cross-library steering performance comparison (TransformerLens, nnsight, repeng, pyvene vs vLLM)
- entry_points: [scripts/bench_external.py]
- depends_on: [core]
- doc: docs/features/external_comparison.md

### analysis
- description: Result aggregation and matplotlib chart generation
- entry_points: [scripts/analyze.py]
- depends_on: [core]
- doc: docs/features/analysis.md
