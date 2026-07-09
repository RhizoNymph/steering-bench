"""``steering-bench`` command-line entry point.

Usage:
  python -m steering_bench run <benchmark> --engine <engine> [common args]
  python -m steering_bench list       # available benchmarks
  python -m steering_bench engines    # discovered engines + capabilities

The ``run`` command drives a single :class:`~steering_bench.harness.Benchmark`
subclass through the engine seam. Comparison benchmarks (``is_comparison``) run
across every discovered engine and print a comparison table.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from typing import Any

from steering_bench.engine.registry import ENGINE_REGISTRY, discover, is_package_available
from steering_bench.engine.serving import SERVING_ENGINE_REGISTRY, get_serving_engine
from steering_bench.harness.args import add_common_args
from steering_bench.harness.benchmark import Benchmark, BenchmarkConfig
from steering_bench.harness.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    PATCH_SWEEP_REGISTRY,
    SERVING_REGISTRY,
    get_benchmark,
    get_patch_sweep_benchmark,
    get_serving_benchmark,
)

PROG = "steering-bench"


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=args.model,
        warmup=args.warmup,
        iters=args.iters,
        max_tokens=args.max_tokens,
        layer=args.layer,
        hook=args.hook,
        output_dir=args.output_dir,
        tag=args.tag,
    )


def _summary_line(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0]


# -- commands -----------------------------------------------------------------


def cmd_list(_: list[str]) -> int:
    print("Available benchmarks:\n")
    for name in sorted(BENCHMARK_REGISTRY):
        cls = BENCHMARK_REGISTRY[name]
        kind = "comparison" if cls.is_comparison else "single-engine"
        print(f"  {name:<22} [{kind}]  {_summary_line(cls.__doc__)}")
    for name in sorted(SERVING_REGISTRY):
        cls = SERVING_REGISTRY[name]
        print(f"  {name:<22} [serving]        {_summary_line(cls.__doc__)}")
    for name in sorted(PATCH_SWEEP_REGISTRY):
        cls = PATCH_SWEEP_REGISTRY[name]
        print(f"  {name:<22} [patch-sweep]    {_summary_line(cls.__doc__)}")
    print()
    return 0


def cmd_engines(_: list[str]) -> int:
    print("Registered engines:\n")
    for entry in ENGINE_REGISTRY:
        available = is_package_available(entry.required_package)
        status = "available" if available else f"{entry.required_package} not installed"
        caps = ", ".join(
            f.name for f in fields(entry.capabilities)
            if getattr(entry.capabilities, f.name)
        ) or "none"
        print(f"  {entry.name:<18} [{status}]")
        print(f"      capabilities: {caps}")
    print("\nDiscovery (importing available adapters):")
    discover()
    return 0


def _print_comparison_table(records: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 78}")
    print("  Engine comparison")
    print(f"{'=' * 78}")
    header = f"{'engine':<18} {'mean_ms':>10} {'p90_ms':>10} {'tok/sec':>10} {'mem_MB':>10}"
    print(header)
    print(f"{'-' * 78}")
    for rec in records:
        if "error" in rec:
            print(f"{rec['engine']:<18} {'FAILED':>10}")
            continue
        res = rec["results"]
        lat = res.get("latency_ms", {})
        print(
            f"{rec['engine']:<18} {lat.get('mean_ms', 0.0):>10.1f} "
            f"{lat.get('p90_ms', 0.0):>10.1f} {res.get('tokens_per_sec', 0.0):>10.0f} "
            f"{res.get('memory_mb', 0.0):>10.0f}"
        )
    print(f"{'=' * 78}\n")


def _run_comparison(
    bench_cls: type[Benchmark],
    config: BenchmarkConfig,
    options: dict[str, Any],
) -> int:
    print("Discovering engines:")
    engine_classes = discover(required=bench_cls.required_capabilities)
    if not engine_classes:
        print("No engines available. Install an engine backend.")
        return 1
    records: list[dict[str, Any]] = []
    for engine_cls in engine_classes:
        print(f"\n--- {engine_cls.name} ---")
        engine = engine_cls()
        try:
            record = bench_cls(engine, config, **options).run()
            records.append({"engine": engine_cls.name, **record})
        except Exception as exc:  # per-engine isolation; report and continue
            print(f"    FAILED: {exc}")
            records.append({"engine": engine_cls.name, "error": str(exc)})
    _print_comparison_table(records)
    return 0


def _run_serving(name: str, rest: list[str]) -> int:
    """Drive a serving benchmark down the ServingEngine async path."""
    bench_cls = get_serving_benchmark(name)
    parser = argparse.ArgumentParser(prog=f"{PROG} run {name}")
    parser.add_argument("benchmark", choices=[name])
    add_common_args(parser)
    bench_cls.add_args(parser)
    args = parser.parse_args(rest)

    serving_names = [e.name for e in SERVING_ENGINE_REGISTRY]
    if args.engine not in serving_names:
        print(
            f"Engine {args.engine!r} has no serving transport; "
            f"serving engines: {', '.join(serving_names)}."
        )
        return 1
    engine = get_serving_engine(args.engine)()
    config = _config_from_args(args)
    options = bench_cls.options_from_args(args)
    bench_cls(engine, config, **options).run()
    return 0


def _run_patch_sweep(name: str, rest: list[str]) -> int:
    """Drive a patch-sweep benchmark down the PatchSweepEngine path.

    Patch sweep is a distinct axis that selects its adapters from ``--variants``
    (``tl_*`` / ``vllm_sweep``), not the steering ``--engine`` flag, so it uses a
    dedicated arg set rather than :func:`add_common_args`.
    """
    bench_cls = get_patch_sweep_benchmark(name)
    parser = argparse.ArgumentParser(prog=f"{PROG} run {name}")
    parser.add_argument("benchmark", choices=[name])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model id.")
    parser.add_argument(
        "--output-dir", default="results/patching", help="Directory for JSON results."
    )
    parser.add_argument("--tag", default="", help="Optional tag recorded in results.")
    bench_cls.add_args(parser)
    args = parser.parse_args(rest)

    config = BenchmarkConfig(
        model=args.model,
        warmup=0,
        iters=0,
        max_tokens=1,
        layer=0,
        hook="post_block",
        output_dir=args.output_dir,
        tag=args.tag,
    )
    options = bench_cls.options_from_args(args)
    bench_cls(config, **options).run()
    return 0


def cmd_run(rest: list[str]) -> int:
    # Peek at the benchmark name (help-agnostic, no choices/errors) so we can
    # register its extra flags before the real parse handles -h and validation.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("benchmark", nargs="?")
    known, _ = peek.parse_known_args(rest)

    if known.benchmark in SERVING_REGISTRY:
        return _run_serving(known.benchmark, rest)
    if known.benchmark in PATCH_SWEEP_REGISTRY:
        return _run_patch_sweep(known.benchmark, rest)

    parser = argparse.ArgumentParser(prog=f"{PROG} run")
    parser.add_argument(
        "benchmark",
        choices=sorted(BENCHMARK_REGISTRY)
        + sorted(SERVING_REGISTRY)
        + sorted(PATCH_SWEEP_REGISTRY),
        help="Benchmark to run.",
    )
    add_common_args(parser)
    if known.benchmark in BENCHMARK_REGISTRY:
        get_benchmark(known.benchmark).add_args(parser)
    args = parser.parse_args(rest)

    bench_cls = get_benchmark(args.benchmark)
    config = _config_from_args(args)
    options = bench_cls.options_from_args(args)

    if bench_cls.is_comparison:
        return _run_comparison(bench_cls, config, options)

    engine_classes = discover(
        filter_names=[args.engine], required=bench_cls.required_capabilities
    )
    if not engine_classes:
        print(
            f"Engine {args.engine!r} is not available or lacks the capabilities "
            f"required by {args.benchmark!r} ({bench_cls.required_capabilities})."
        )
        return 1
    engine = engine_classes[0]()
    bench_cls(engine, config, **options).run()
    return 0


def _top_usage() -> None:
    print(
        f"usage: {PROG} <command> [args]\n\n"
        "commands:\n"
        "  run <benchmark> --engine <engine> [args]   run a benchmark\n"
        "  list                                       list available benchmarks\n"
        "  engines                                    list engines + capabilities\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        _top_usage()
        return 2
    command, rest = args[0], args[1:]
    match command:
        case "list":
            return cmd_list(rest)
        case "engines":
            return cmd_engines(rest)
        case "run":
            return cmd_run(rest)
        case "-h" | "--help":
            _top_usage()
            return 0
        case _:
            print(f"unknown command {command!r}\n")
            _top_usage()
            return 2


if __name__ == "__main__":
    sys.exit(main())
