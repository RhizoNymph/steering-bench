"""Batched external-comparison benchmark tests (no GPU / no backends)."""

from __future__ import annotations

import argparse
from pathlib import Path

from steering_bench.engine.spec import SteeringSpec
from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.harness.benchmarks.external_comparison import (
    ExternalComparisonBenchmark,
)
from steering_bench.timing import compute_stats
from tests.test_harness_benchmark import FakeEngine

# Offline model (present in MODEL_CONFIGS static table).
_MODEL = "meta-llama/Llama-3.2-1B"


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=_MODEL,
        warmup=1,
        iters=2,
        max_tokens=8,
        layer=8,
        hook="post_block",
        output_dir=str(tmp_path),
    )


def _bench(tmp_path: Path, **options: object) -> ExternalComparisonBenchmark:
    return ExternalComparisonBenchmark(FakeEngine(), _config(tmp_path), **options)


def test_default_batch_size_is_one(tmp_path: Path) -> None:
    bench = _bench(tmp_path)
    requests = bench.build_requests()
    assert len(requests) == 1
    assert bench.parameters()["batch_size"] == 1


def test_batched_builds_n_requests(tmp_path: Path) -> None:
    bench = _bench(tmp_path, batch_size=16)
    requests = bench.build_requests()
    assert len(requests) == 16
    # Every request carries the same inline steering spec.
    assert all(isinstance(r.steering, SteeringSpec) for r in requests)
    assert bench.parameters()["batch_size"] == 16


def test_add_args_and_options_from_args() -> None:
    parser = argparse.ArgumentParser()
    ExternalComparisonBenchmark.add_args(parser)
    args = parser.parse_args(["--batch-size", "8", "--prompt-len", "32"])
    opts = ExternalComparisonBenchmark.options_from_args(args)
    assert opts == {"batch_size": 8, "prompt_len": 32}


def test_extra_results_tier2_metrics(tmp_path: Path) -> None:
    bench = _bench(tmp_path, batch_size=16)
    stats = compute_stats([100.0, 100.0])  # mean 100 ms
    extra = bench.extra_results(stats, avg_output_tokens=8.0, num_requests=16)
    assert extra["req_per_sec"] == 16 / (100.0 / 1000.0)
    assert extra["avg_per_request_ms"] == 100.0 / 16


def test_extra_results_empty_for_single(tmp_path: Path) -> None:
    bench = _bench(tmp_path, batch_size=1)
    stats = compute_stats([100.0])
    assert bench.extra_results(stats, avg_output_tokens=8.0, num_requests=1) == {}


def test_batched_run_end_to_end(tmp_path: Path) -> None:
    bench = _bench(tmp_path, batch_size=4)
    record = bench.run()
    assert record["parameters"]["batch_size"] == 4
    assert record["results"]["num_requests"] == 4
    assert "req_per_sec" in record["results"]
