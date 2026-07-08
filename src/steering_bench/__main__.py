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
from steering_bench.harness.args import add_common_args
from steering_bench.harness.benchmark import Benchmark, BenchmarkConfig
from steering_bench.harness.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark

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
    engine_classes = discover()
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


def cmd_run(rest: list[str]) -> int:
    # Peek at the benchmark name (help-agnostic, no choices/errors) so we can
    # register its extra flags before the real parse handles -h and validation.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("benchmark", nargs="?")
    known, _ = peek.parse_known_args(rest)

    parser = argparse.ArgumentParser(prog=f"{PROG} run")
    parser.add_argument(
        "benchmark",
        choices=sorted(BENCHMARK_REGISTRY),
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

    engine_classes = discover(filter_names=[args.engine])
    if not engine_classes:
        print(f"Engine {args.engine!r} is not available in this environment.")
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
