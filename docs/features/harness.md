# Benchmark Harness

The harness kills the per-script boilerplate that every benchmark used to
repeat — the duplicated model-dimension table, the copy-pasted argparse block,
and the hand-rolled warmup/measure loop — and gives the repo a single entry
point. It is built on the Phase A engine seam (`steering_bench.engine`): a
feature is a small `Benchmark` subclass and it runs against any engine the
registry can discover.

## Scope

### In Scope (Phase B)
- `get_model_config(model_id) -> ModelConfig` — single source of truth for
  `hidden_size` / `num_layers`: a static table for known models, HuggingFace
  `AutoConfig` fallback for anything else (transformers imported lazily).
- `add_common_args(parser)` — the argparse flags shared across benchmarks
  (`--model`, `--output-dir`, `--warmup`, `--iters`, `--max-tokens`, `--layer`,
  `--hook`, `--tag`, `--engine`), composable with per-script extra flags.
- `Benchmark` base class owning `load -> warmup -> measure -> teardown ->
  write -> print`; a subclass supplies the workload (`build_requests`), its
  `parameters` block, and a `benchmark_name`.
- A `steering-bench` CLI (`python -m steering_bench`) with a benchmark registry
  and `run` / `list` / `engines` commands, plus two proof benchmarks
  (`latency`, `external-comparison`) built entirely on the engine seam.
- Additive: no existing script, module, or test is modified.

### Not In Scope
- Migrating the existing `scripts/*.py` onto the harness (dropping their
  `sys.path.insert`, local `MODEL_CONFIGS`, and hand-rolled loops) — that is
  **Phase C**, along with a first-class generic `engine` result block.
- New engines beyond those already in `engine.registry`.
- Throughput/memory/ablation benchmarks — only latency + external-comparison
  ship as proof; more benchmarks are follow-ups.

## Data / Control Flow

### `Benchmark.run` (single engine)
```
Benchmark(engine, config, **options).run()
  requests = self.build_requests()          # subclass workload
  engine.load(config.model, **self.load_opts())
  try:
    memory_mb = engine.memory_allocated_mb()
    for _ in range(config.warmup):          # unmeasured
        engine.generate(requests)
    for _ in range(config.iters):           # measured
        _sync()                             # torch.cuda.synchronize() iff CUDA
        t0 = perf_counter()
        results = engine.generate(requests)
        _sync(); t1 = perf_counter()
        samples_ms.append((t1 - t0) * 1000)
        output_tokens_per_iter.append(sum(r.output_tokens for r in results))
  finally:
    engine.teardown()                       # always frees the engine
  stats = compute_stats(samples_ms)         # steering_bench.timing
  tokens_per_sec = avg_output_tokens / (stats.mean_ms / 1000)
  write_result(benchmark_name, parameters(), results_block, output_dir, tag, raw_samples_ms)
  print_result_summary(benchmark_name, results_block)
  return {benchmark, parameters, results, output_path}
```
`_sync()` is a no-op on CPU-only hosts, so the loop (and its tests) run without
a GPU. `teardown` is in a `finally`, so the engine is freed even if `generate`
raises.

### CLI (`python -m steering_bench`)
```
main(argv)
  list     -> cmd_list      : iterate BENCHMARK_REGISTRY, print name/kind/summary
  engines  -> cmd_engines   : print ENGINE_REGISTRY entries + capabilities,
                              then engine.registry.discover() (prints availability)
  run B    -> cmd_run:
      parser = common args + choices=BENCHMARK_REGISTRY
      peek benchmark -> bench_cls.add_args(parser) -> reparse   (per-benchmark flags)
      config  = BenchmarkConfig(from args)
      options = bench_cls.options_from_args(args)
      if bench_cls.is_comparison:                 # external-comparison
          for engine_cls in discover():           # every available engine
              record = bench_cls(engine_cls(), config, **options).run()
          _print_comparison_table(records)
      else:                                        # latency
          [engine_cls] = discover(filter_names=[args.engine])
          bench_cls(engine_cls(), config, **options).run()
```

### `get_model_config`
```
get_model_config(model_id)
  MODEL_CONFIGS[model_id]?              -> ModelConfig                (offline)
  else import transformers (lazy); AutoConfig.from_pretrained(model_id)
      hidden_size <- first of (hidden_size, n_embd, d_model, hidden_dim)
      num_layers  <- first of (num_hidden_layers, n_layer, num_layers, n_layers)
      missing either -> ModelConfigError with the tried names + model id
```

## Related Files

| File | Role | Key exports |
|------|------|-------------|
| `src/steering_bench/harness/models.py` | Model-dimension resolution | `ModelConfig`, `MODEL_CONFIGS`, `get_model_config`, `ModelConfigError` |
| `src/steering_bench/harness/args.py` | Shared argparse flags | `add_common_args`, `engine_names`, `DEFAULT_*` |
| `src/steering_bench/harness/benchmark.py` | Lifecycle base class | `Benchmark` (`build_requests`, `parameters`, `load_opts`, `extra_results`, `add_args`, `options_from_args`, `run`, `is_comparison`), `BenchmarkConfig` |
| `src/steering_bench/harness/benchmarks/workload.py` | Workload builders | `steering_spec_for`, `make_prompt`, `steered_requests` |
| `src/steering_bench/harness/benchmarks/latency.py` | Latency benchmark | `LatencyBenchmark` |
| `src/steering_bench/harness/benchmarks/external_comparison.py` | Cross-engine comparison | `ExternalComparisonBenchmark` |
| `src/steering_bench/harness/benchmarks/registry.py` | Benchmark registry | `BENCHMARK_REGISTRY`, `get_benchmark` |
| `src/steering_bench/harness/__init__.py` | Public re-exports | (surface above) |
| `src/steering_bench/__main__.py` | CLI | `main`, `cmd_run`, `cmd_list`, `cmd_engines` |
| `pyproject.toml` `[project.scripts]` | Console entry point | `steering-bench = steering_bench.__main__:main` |
| `src/steering_bench/timing.py`, `output.py` | Reused (not duplicated) | `compute_stats`, `write_result`, `print_result_summary` |
| `tests/test_harness_models.py`, `test_harness_args.py`, `test_harness_benchmark.py` | Unit tests (no GPU/backends) | — |

## Invariants / Constraints
- **No vLLM/backends at module scope.** The harness goes entirely through the
  `SteeringEngine` seam; `transformers` in `models.py` is imported lazily. The
  package (and its tests) import and run with no engine backend and no
  transformers installed.
- **CPU-safe measure loop.** GPU synchronization only happens when
  `torch.cuda.is_available()`; otherwise the loop still runs (used by the
  fake-engine tests).
- **Teardown always runs.** `engine.teardown()` is in a `finally`, so a failing
  `generate` still frees the engine.
- **Static table is authoritative for known models**; the AutoConfig fallback is
  only consulted on a miss, and raises `ModelConfigError` (a `ValueError`) with
  an actionable message if it cannot find both dimensions.
- **`--engine` choices come from the engine registry**, so the CLI cannot select
  an engine that does not exist. Comparison benchmarks ignore `--engine` and run
  every discovered engine.
- **Additive.** Nothing in `scripts/`, `external/`, or the existing tests is
  modified; migrating those onto the harness is Phase C.
