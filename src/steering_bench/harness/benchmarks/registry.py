"""Benchmark registry: benchmark name -> ``Benchmark`` subclass."""

from __future__ import annotations

from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.capture import CaptureBenchmark
from steering_bench.harness.benchmarks.external_comparison import (
    ExternalComparisonBenchmark,
)
from steering_bench.harness.benchmarks.latency import LatencyBenchmark
from steering_bench.harness.benchmarks.patch_sweep import PatchSweepBenchmark
from steering_bench.harness.benchmarks.serving import ServingBenchmark
from steering_bench.harness.benchmarks.throughput import ThroughputBenchmark

BENCHMARK_REGISTRY: dict[str, type[Benchmark]] = {
    "latency": LatencyBenchmark,
    "throughput": ThroughputBenchmark,
    "external-comparison": ExternalComparisonBenchmark,
    "capture": CaptureBenchmark,
}

# Serving benchmarks drive the ServingEngine seam (online/HTTP transport), not
# the synchronous ``Benchmark`` measure-loop, so they live in a separate registry
# the CLI dispatches down a parallel async path.
SERVING_REGISTRY: dict[str, type[ServingBenchmark]] = {
    "serving": ServingBenchmark,
}

# Patch-sweep benchmarks drive the PatchSweepEngine seam (activation-patching /
# causal tracing) -- a third axis with a distinct result shape (cells / wall_s /
# argmax) that spans two engine types per run, so they live in their own registry
# the CLI dispatches down a parallel path.
PATCH_SWEEP_REGISTRY: dict[str, type[PatchSweepBenchmark]] = {
    "patch-sweep": PatchSweepBenchmark,
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


def get_serving_benchmark(name: str) -> type[ServingBenchmark]:
    """Look up a serving benchmark class by name, raising ``KeyError`` if unknown."""
    try:
        return SERVING_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown serving benchmark {name!r}; available: "
            f"{', '.join(sorted(SERVING_REGISTRY))}"
        ) from None


def get_patch_sweep_benchmark(name: str) -> type[PatchSweepBenchmark]:
    """Look up a patch-sweep benchmark class by name, raising ``KeyError`` if unknown."""
    try:
        return PATCH_SWEEP_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown patch-sweep benchmark {name!r}; available: "
            f"{', '.join(sorted(PATCH_SWEEP_REGISTRY))}"
        ) from None
