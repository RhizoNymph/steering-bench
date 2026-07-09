"""Benchmark registry: benchmark name -> ``Benchmark`` subclass."""

from __future__ import annotations

from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.external_comparison import (
    ExternalComparisonBenchmark,
)
from steering_bench.harness.benchmarks.latency import LatencyBenchmark
from steering_bench.harness.benchmarks.throughput import ThroughputBenchmark

BENCHMARK_REGISTRY: dict[str, type[Benchmark]] = {
    "latency": LatencyBenchmark,
    "throughput": ThroughputBenchmark,
    "external-comparison": ExternalComparisonBenchmark,
}


def get_benchmark(name: str) -> type[Benchmark]:
    """Look up a benchmark class by name, raising ``KeyError`` if unknown."""
    try:
        return BENCHMARK_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; available: "
            f"{', '.join(sorted(BENCHMARK_REGISTRY))}"
        ) from None
