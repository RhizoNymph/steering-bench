# Activation Steering Benchmark Suite

A standalone benchmark harness that compares vLLM's activation steering against TransformerLens, nnsight, repeng, and pyvene. Also measures vLLM-internal overhead and optimization interactions.

Completely external to the vLLM codebase — vLLM is an optional dependency.

## Project Structure

```
steering-bench/
  pyproject.toml
  README.md

  src/steering_bench/
    __init__.py
    cli.py                               # Click CLI: steering-bench run|analyze|plot|env|list
    config.py                            # BenchmarkConfig dataclass, parameter sweeps
    vectors.py                           # Steering vector generation (random seeded, repeng)
    timing.py                            # GPU-synced timing context manager, stats aggregation
    output.py                            # JSON result schema, writer, environment capture
    models.py                            # Model loading helpers, shared tokenizer, prompt gen

    micro/                               # Microbenchmarks (vLLM internals, no model serving)
      __init__.py
      steering_op.py                     # torch.ops.vllm.apply_steering kernel timing
      steering_manager.py                # SteeringManager method timing
      index_building.py                  # steering_index construction loop

    vllm_bench/                          # vLLM system benchmarks (steering on vs off)
      __init__.py
      latency.py                         # Single-batch latency via vllm.LLM Python API
      throughput.py                      # Offline batch throughput
      serving.py                         # Online serving via HTTP (vllm serve subprocess)
      memory.py                          # GPU memory delta measurement
      startup.py                         # Startup time with/without steering

    ablation/                            # Optimization interaction tests
      __init__.py
      cuda_graphs.py                     # enforce_eager True/False x steering True/False
      compile_level.py                   # compilation_config.level 0/3 x steering
      prefix_cache.py                    # Prefix caching interaction with steering
      config_scaling.py                  # max_steering_configs: 1, 2, 4, 8, 16, 32
      hook_points.py                     # 1 vs 2 vs 3 active hooks per layer
      model_family.py                    # Same benchmark across Gemma3, LLaMA, Qwen

    external/                            # Cross-library comparison
      __init__.py
      base.py                            # SteeringBenchmark protocol
      hf_baseline.py                     # HuggingFace generate(), no steering
      transformerlens_bench.py           # TransformerLens HookedTransformer
      nnsight_bench.py                   # nnsight LanguageModel + trace
      repeng_bench.py                    # repeng ControlVector
      pyvene_bench.py                    # pyvene IntervenableModel
      vllm_single.py                     # vLLM single-request (apples-to-apples)
      vllm_batched.py                    # vLLM continuous batching (what batching buys you)

    analysis/                            # Post-run analysis and visualization
      __init__.py
      aggregate.py                       # Merge JSON results into DataFrame
      plot_overhead.py                   # Steering on/off overhead charts
      plot_comparison.py                 # Cross-library comparison
      plot_ablation.py                   # Ablation heatmaps, tornado charts
      plot_scaling.py                    # Config count / batch size curves

  scripts/
    run_micro.sh
    run_vllm.sh
    run_ablation.sh
    run_external.sh
    run_all.sh
    run_parallel.sh                      # 3-machine SSH dispatch

  results/                               # Gitignored, created at runtime
    .gitkeep

  tests/
    test_config.py
    test_vectors.py
    test_timing.py
    test_output.py
```

## Dependencies

```toml
[project]
name = "steering-bench"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
    "click",
    "rich",
]

[project.optional-dependencies]
vllm = ["vllm"]
transformerlens = ["transformer-lens>=2.0"]
nnsight = ["nnsight>=0.3"]
repeng = ["repeng>=0.3"]
pyvene = ["pyvene>=0.1"]
all-external = [
    "transformer-lens>=2.0",
    "nnsight>=0.3",
    "repeng>=0.3",
    "pyvene>=0.1",
]
all = ["steering-bench[vllm,all-external]"]
dev = ["pytest", "ruff", "mypy"]

[project.scripts]
steering-bench = "steering_bench.cli:main"
```

Each library is optional. Benchmarks that require an unavailable library skip gracefully. You can install just what you need:

```bash
# Everything
pip install -e ".[all]"

# Just vLLM internal benchmarks
pip install -e ".[vllm]"

# Just external comparison (no vLLM)
pip install -e ".[all-external]"

# Point at a local vLLM fork
pip install -e /path/to/vllm && pip install -e "."
```

## CLI

```bash
steering-bench run --all --model meta-llama/Llama-3.2-1B --output-dir results/
steering-bench run micro
steering-bench run vllm --model google/gemma-3-4b-it
steering-bench run ablation --model google/gemma-3-4b-it
steering-bench run external --model meta-llama/Llama-3.2-1B
steering-bench run external.repeng --model meta-llama/Llama-3.2-1B

steering-bench analyze --results-dir results/
steering-bench plot --results-dir results/ --output-dir results/plots/
steering-bench env
steering-bench list
```

**Common flags:** `--model`, `--output-dir`, `--warmup N`, `--iters N`, `--seed N` (default 42), `--vector-scale F` (default 0.1), `--gpu-id N`, `--pin-clocks`, `--tag TEXT`

## Hardware & Models

**Available:** 3x RTX 3090 (24GB), separate machines. Willing to rent A100/H100 for larger models.

| Purpose | Model | Why |
|---------|-------|-----|
| Cross-library comparison | `meta-llama/Llama-3.2-1B` | All 4 external libs support LLaMA |
| vLLM internal (primary) | `google/gemma-3-4b-it` | Most mature steering integration |
| vLLM internal (secondary) | `meta-llama/Llama-3.2-1B` | Cross-family consistency |
| vLLM internal (large) | `meta-llama/Llama-3.1-8B` | Production-scale (needs rented GPU) |

**Vectors:** Random (seed=42, scale=0.1) for performance. One repeng-generated vector for sanity validation.

## Output Schema

Every benchmark writes JSON:

```json
{
  "benchmark": "external.repeng",
  "timestamp": "2026-04-08T12:00:00Z",
  "tag": "canonical-run-1",
  "environment": {
    "gpu": "NVIDIA RTX 3090",
    "gpu_uuid": "GPU-xxxxx",
    "cuda_version": "12.x",
    "torch_version": "2.x",
    "vllm_version": "0.x.y+abc123",
    "python_version": "3.12",
    "hostname": "machine-a",
    "clocks_pinned": true
  },
  "parameters": { ... },
  "results": {
    "latency_ms": {
      "mean": 45.2, "median": 44.8,
      "p10": 42.1, "p90": 48.3, "p99": 52.7,
      "stddev": 2.1, "n": 20
    },
    "throughput_tokens_per_sec": 1137.2,
    "memory_peak_mb": 2340.5
  },
  "raw_samples_ms": [44.1, 45.2, ...]
}
```

## 10 Questions This Suite Answers

1. What is the per-token overhead of the gather+add steering op?
2. What is the per-step Python overhead of table population + index building?
3. What is the end-to-end latency impact at various batch sizes?
4. What is the throughput impact under realistic serving load?
5. How much GPU memory do steering buffers cost?
6. Does steering break CUDA graph benefits?
7. Does torch.compile interact badly with the opaque steering op?
8. How does overhead scale with number of concurrent configs?
9. How does vLLM compare to external libs for single-request steering?
10. How much does continuous batching help for multi-config workloads?
