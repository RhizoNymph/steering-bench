"""External-comparison benchmark: one steered request across every engine.

The engine-agnostic essence of ``bench_external.py`` Tier 1: run the same
single-request steered workload through each discovered engine and let the CLI
tabulate the results side by side. Each engine still runs through the ordinary
:meth:`Benchmark.run` lifecycle; ``is_comparison`` tells the CLI to iterate
engines and print a comparison table.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.spec import GenerationRequest
from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.workload import steered_requests


class ExternalComparisonBenchmark(Benchmark):
    """Single-request steered latency, run across engines for comparison."""

    benchmark_name = "harness.external_comparison"
    is_comparison = True

    def build_requests(self) -> list[GenerationRequest]:
        return steered_requests(
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
            max_tokens=self.config.max_tokens,
            batch_size=1,
        )

    def parameters(self) -> dict[str, Any]:
        params = super().parameters()
        params["batch_size"] = 1
        return params
