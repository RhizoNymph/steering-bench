"""Harness measure-loop tests driven by a fake engine (no GPU, no backends)."""

from __future__ import annotations

import json
from pathlib import Path

from steering_bench.engine.base import Capabilities, SteeringEngine
from steering_bench.engine.spec import GenerationRequest, GenerationResult
from steering_bench.harness.benchmark import Benchmark, BenchmarkConfig


class FakeEngine(SteeringEngine):
    """In-memory engine that records lifecycle calls and returns canned results."""

    name = "fake"
    capabilities = Capabilities(batching=True)

    def __init__(self, tokens_per_request: int = 7) -> None:
        self.tokens_per_request = tokens_per_request
        self.load_calls = 0
        self.generate_calls = 0
        self.teardown_calls = 0
        self.last_opts: dict[str, object] = {}

    def load(self, model_id: str, **opts: object) -> None:
        self.load_calls += 1
        self.loaded_model = model_id
        self.last_opts = dict(opts)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        self.generate_calls += 1
        return [
            GenerationResult(output_tokens=self.tokens_per_request) for _ in requests
        ]

    def memory_allocated_mb(self) -> float:
        return 123.5

    def teardown(self) -> None:
        self.teardown_calls += 1


class _TwoRequestBenchmark(Benchmark):
    benchmark_name = "test.two_request"

    def build_requests(self) -> list[GenerationRequest]:
        return [
            GenerationRequest(prompt="hello", max_tokens=self.config.max_tokens),
            GenerationRequest(prompt="world", max_tokens=self.config.max_tokens),
        ]

    def parameters(self) -> dict[str, object]:
        params = super().parameters()
        params["workload"] = "two_request"
        return params


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        model="fake/model",
        warmup=3,
        iters=5,
        max_tokens=16,
        layer=8,
        hook="post_block",
        output_dir=str(tmp_path),
        tag="unit",
    )


def test_run_lifecycle_call_counts(tmp_path: Path) -> None:
    engine = FakeEngine()
    bench = _TwoRequestBenchmark(engine, _config(tmp_path))
    bench.run()

    assert engine.load_calls == 1
    # warmup + iters generate() invocations.
    assert engine.generate_calls == 3 + 5
    assert engine.teardown_calls == 1


def test_run_returns_result_dict(tmp_path: Path) -> None:
    engine = FakeEngine()
    bench = _TwoRequestBenchmark(engine, _config(tmp_path))
    record = bench.run()

    assert record["benchmark"] == "test.two_request"
    assert "parameters" in record
    assert "results" in record
    params = record["parameters"]
    assert params["model"] == "fake/model"
    assert params["engine"] == "fake"
    assert params["workload"] == "two_request"
    # Derived throughput metrics are present.
    assert "tokens_per_sec" in record["results"]
    assert "memory_mb" in record["results"]
    assert record["results"]["memory_mb"] == 123.5


def test_run_writes_json(tmp_path: Path) -> None:
    engine = FakeEngine()
    bench = _TwoRequestBenchmark(engine, _config(tmp_path))
    record = bench.run()

    written = list(Path(tmp_path).glob("*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text())
    assert data["benchmark"] == "test.two_request"
    assert data["tag"] == "unit"
    assert data["parameters"]["workload"] == "two_request"
    assert "results" in data
    assert str(written[0]) == record["output_path"]


def test_teardown_called_even_on_generate_error(tmp_path: Path) -> None:
    class _BoomEngine(FakeEngine):
        def generate(self, requests):  # type: ignore[no-untyped-def]
            self.generate_calls += 1
            raise RuntimeError("boom")

    engine = _BoomEngine()
    bench = _TwoRequestBenchmark(engine, _config(tmp_path))
    try:
        bench.run()
    except RuntimeError:
        pass
    assert engine.teardown_calls == 1
