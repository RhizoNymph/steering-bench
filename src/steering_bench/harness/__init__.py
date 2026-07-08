"""Benchmark harness: shared model config, args, and the ``Benchmark`` lifecycle.

Public surface:
  * ``get_model_config`` / ``ModelConfig`` -- single source of truth for model
    dimensions (static table + HuggingFace AutoConfig fallback).
  * ``add_common_args`` -- shared argparse flags for benchmark scripts/CLI.
  * ``Benchmark`` / ``BenchmarkConfig`` -- the warmup->measure->write->print
    lifecycle a feature subclasses.
  * ``BENCHMARK_REGISTRY`` / ``get_benchmark`` -- name -> ``Benchmark`` subclass.

Built on the engine seam in :mod:`steering_bench.engine`.
"""

from __future__ import annotations

from steering_bench.harness.args import add_common_args, engine_names
from steering_bench.harness.benchmark import Benchmark, BenchmarkConfig
from steering_bench.harness.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    get_benchmark,
)
from steering_bench.harness.models import (
    MODEL_CONFIGS,
    ModelConfig,
    ModelConfigError,
    get_model_config,
)

__all__ = [
    "BENCHMARK_REGISTRY",
    "Benchmark",
    "BenchmarkConfig",
    "MODEL_CONFIGS",
    "ModelConfig",
    "ModelConfigError",
    "add_common_args",
    "engine_names",
    "get_benchmark",
    "get_model_config",
]
