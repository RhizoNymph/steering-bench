"""Latency benchmark: generation latency at a given layer/hook, any engine.

The engine-agnostic essence of ``scripts/bench_latency.py``: build a batch of
identically-steered requests and let :meth:`Benchmark.run` time them.
"""

from __future__ import annotations

import argparse
from typing import Any

from steering_bench.engine.spec import GenerationRequest
from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.workload import steered_requests

DEFAULT_BATCH_SIZE = 1
DEFAULT_PROMPT_LEN = 64


class LatencyBenchmark(Benchmark):
    """Single/batched steered-generation latency for the chosen engine."""

    benchmark_name = "harness.latency"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Number of requests generated per measured iteration.",
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

    def build_requests(self) -> list[GenerationRequest]:
        return steered_requests(
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
            max_tokens=self.config.max_tokens,
            batch_size=int(self.options.get("batch_size", DEFAULT_BATCH_SIZE)),
            prompt_len=int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN)),
        )

    def parameters(self) -> dict[str, Any]:
        params = super().parameters()
        params["batch_size"] = int(self.options.get("batch_size", DEFAULT_BATCH_SIZE))
        params["prompt_len"] = int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN))
        return params
