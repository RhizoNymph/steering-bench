"""External-comparison benchmark: a steered workload across every engine.

The engine-agnostic essence of ``bench_external.py``: run the same steered
workload through each discovered engine and let the CLI tabulate the results
side by side.  ``--batch-size 1`` (the default) is the Tier-1 single-request
comparison; ``--batch-size N`` is the Tier-2 batched comparison, adding
``req_per_sec`` / ``avg_per_request_ms`` (the metrics ``bench_external.run_tier2``
produced) to the result.  Each engine still runs through the ordinary
:meth:`Benchmark.run` lifecycle; ``is_comparison`` tells the CLI to iterate
engines and print a comparison table.
"""

from __future__ import annotations

import argparse
from typing import Any

from steering_bench.engine.spec import GenerationRequest
from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.workload import steered_requests
from steering_bench.timing import TimingStats

DEFAULT_BATCH_SIZE = 1
DEFAULT_PROMPT_LEN = 64


class ExternalComparisonBenchmark(Benchmark):
    """Single/batched steered latency, run across engines for comparison."""

    benchmark_name = "harness.external_comparison"
    is_comparison = True

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Requests per measured iteration (1 = Tier-1, N = Tier-2 batched).",
        )
        parser.add_argument(
            "--prompt-len",
            type=int,
            default=DEFAULT_PROMPT_LEN,
            help="Approximate prompt length in tokens.",
        )

    @staticmethod
    def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
        return {"batch_size": args.batch_size, "prompt_len": args.prompt_len}

    def _batch_size(self) -> int:
        return int(self.options.get("batch_size", DEFAULT_BATCH_SIZE))

    def build_requests(self) -> list[GenerationRequest]:
        return steered_requests(
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
            max_tokens=self.config.max_tokens,
            batch_size=self._batch_size(),
            prompt_len=int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN)),
        )

    def parameters(self) -> dict[str, Any]:
        params = super().parameters()
        params["batch_size"] = self._batch_size()
        params["prompt_len"] = int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN))
        return params

    def extra_results(
        self, stats: TimingStats, avg_output_tokens: float, num_requests: int
    ) -> dict[str, Any]:
        # Tier-2 batched throughput metrics, mirroring bench_external.run_tier2.
        if num_requests <= 1 or stats.mean_ms <= 0:
            return {}
        return {
            "req_per_sec": num_requests / (stats.mean_ms / 1000.0),
            "avg_per_request_ms": stats.mean_ms / num_requests,
        }
