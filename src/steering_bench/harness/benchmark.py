"""The ``Benchmark`` base class: warmup -> measure -> write -> print.

A feature becomes a small ``Benchmark`` subclass that supplies a workload
(``build_requests``), its ``parameters`` block, and a ``benchmark_name``. The
base owns the lifecycle: it loads the engine, runs unmeasured warmup iterations,
times ``iters`` measured iterations (GPU-synced when CUDA is available), computes
statistics via :func:`steering_bench.timing.compute_stats`, tears the engine
down, and writes/prints the result through :mod:`steering_bench.output`.

Nothing here is vLLM-specific: the base goes entirely through the
``SteeringEngine`` seam, so any engine and any workload compose.
"""

from __future__ import annotations

import abc
import argparse
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from steering_bench.engine.base import SteeringConfig, SteeringEngine
from steering_bench.engine.spec import GenerationRequest
from steering_bench.output import print_result_summary, write_result
from steering_bench.timing import TimingStats, compute_stats


@dataclass(frozen=True)
class BenchmarkConfig:
    """Common configuration shared by every benchmark run."""

    model: str
    warmup: int
    iters: int
    max_tokens: int
    layer: int
    hook: str
    output_dir: str
    tag: str = ""


def _sync() -> None:
    """GPU barrier when CUDA is present; a no-op on CPU-only hosts."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Benchmark(abc.ABC):
    """Base class owning the warmup/measure/write/print lifecycle.

    Subclasses set ``benchmark_name`` and implement ``build_requests``. They may
    extend ``parameters`` (call ``super().parameters()``), supply engine
    ``load_opts``, or add ``extra_results`` derived metrics.
    """

    benchmark_name: ClassVar[str] = "benchmark"
    #: When True, the CLI runs this benchmark across every discovered engine and
    #: tabulates a comparison rather than running a single ``--engine``.
    is_comparison: ClassVar[bool] = False

    def __init__(
        self, engine: SteeringEngine, config: BenchmarkConfig, **options: Any
    ) -> None:
        self.engine = engine
        self.config = config
        self.options = options

    # -- CLI extension hooks (per-benchmark extra flags) ---------------------

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Register benchmark-specific flags. Default: none."""

    @staticmethod
    def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
        """Collect benchmark-specific options from parsed args. Default: none."""
        return {}

    # -- subclass hooks ------------------------------------------------------

    @abc.abstractmethod
    def build_requests(self) -> list[GenerationRequest]:
        """The workload: one batch of requests measured each iteration."""

    def parameters(self) -> dict[str, Any]:
        """Result ``parameters`` block. Subclasses extend via ``super()``."""
        return {
            "model": self.config.model,
            "engine": self.engine.name,
            "layer": self.config.layer,
            "hook": self.config.hook,
            "max_tokens": self.config.max_tokens,
            "warmup": self.config.warmup,
            "iters": self.config.iters,
        }

    def load_opts(self) -> dict[str, Any]:
        """Engine-specific ``load`` options. Default: none."""
        return {}

    def steering_config(self) -> SteeringConfig | None:
        """Typed load-time steering config. Default: ``None`` (engine default)."""
        return None

    def after_load(self) -> None:
        """Hook run once after ``load`` and before warmup. Default: no-op.

        Used by mode benchmarks to register named steering modules.
        """

    def extra_results(
        self, stats: TimingStats, avg_output_tokens: float, num_requests: int
    ) -> dict[str, Any]:
        """Extra derived metrics merged into the ``results`` block."""
        return {}

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full benchmark and return the written record dict."""
        requests = self.build_requests()
        num_requests = len(requests)

        steering_config = self.steering_config()
        load_opts = self.load_opts()
        if steering_config is not None:
            self.engine.load(
                self.config.model, steering_config=steering_config, **load_opts
            )
        else:
            self.engine.load(self.config.model, **load_opts)
        try:
            self.after_load()
            memory_mb = self.engine.memory_allocated_mb()

            for _ in range(self.config.warmup):
                self.engine.generate(requests)

            samples_ms: list[float] = []
            output_tokens_per_iter: list[int] = []
            for _ in range(self.config.iters):
                _sync()
                t0 = time.perf_counter()
                results = self.engine.generate(requests)
                _sync()
                t1 = time.perf_counter()
                samples_ms.append((t1 - t0) * 1000.0)
                output_tokens_per_iter.append(
                    sum(r.output_tokens for r in results)
                )
        finally:
            self.engine.teardown()

        stats = compute_stats(samples_ms)
        avg_output_tokens = (
            sum(output_tokens_per_iter) / len(output_tokens_per_iter)
            if output_tokens_per_iter
            else 0.0
        )
        tokens_per_sec = (
            avg_output_tokens / (stats.mean_ms / 1000.0) if stats.mean_ms > 0 else 0.0
        )

        results_block: dict[str, Any] = {
            "latency_ms": {
                k: v for k, v in stats.to_dict().items() if k != "samples_ms"
            },
            "memory_mb": memory_mb,
            "avg_output_tokens": avg_output_tokens,
            "tokens_per_sec": tokens_per_sec,
            "num_requests": num_requests,
        }
        results_block.update(
            self.extra_results(stats, avg_output_tokens, num_requests)
        )

        params = self.parameters()
        path = write_result(
            benchmark=self.benchmark_name,
            parameters=params,
            results=results_block,
            output_dir=self.config.output_dir,
            tag=self.config.tag,
            raw_samples_ms=stats.samples_ms,
            engine=self.engine.identity(),
        )
        print_result_summary(self.benchmark_name, results_block)

        return {
            "benchmark": self.benchmark_name,
            "parameters": params,
            "results": results_block,
            "output_path": str(path),
        }
