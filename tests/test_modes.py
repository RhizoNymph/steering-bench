"""Tests for the engine-agnostic steering mode catalog (Phase 3)."""

from __future__ import annotations

import pytest

from steering_bench.engine.base import Capabilities, SteeringConfig, SteeringEngine
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    SteeringSpec,
)
from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.harness.benchmarks.latency import LatencyBenchmark
from steering_bench.harness.benchmarks.modes import (
    NAMED_MODULE,
    ModeError,
    build_mode_requests,
    max_steering_configs_for,
    mode_enable_steering,
    mode_needs_named_module,
    parse_mode,
    steering_config_for,
)

MODEL = "facebook/opt-125m"  # in the static table: hidden_size=768, num_layers=12
KW = {"model": MODEL, "layer": 3, "hook": "post_block", "max_tokens": 8, "prompt_len": 8}


# -- parse_mode / classifiers ------------------------------------------------


def test_parse_mode_base() -> None:
    assert parse_mode("disabled") == ("disabled", None)
    assert parse_mode("inline_shared") == ("inline_shared", None)


def test_parse_mode_per_request() -> None:
    assert parse_mode("per_request_4") == ("per_request", 4)
    assert parse_mode("per_request_1") == ("per_request", 1)


def test_parse_mode_unknown() -> None:
    with pytest.raises(ModeError):
        parse_mode("bogus")
    with pytest.raises(ModeError):
        parse_mode("per_request_0")
    with pytest.raises(ModeError):
        parse_mode("per_request_x")


def test_enable_steering() -> None:
    assert mode_enable_steering("disabled") is False
    assert mode_enable_steering("enabled_idle") is True
    assert mode_enable_steering("named_shared") is True


def test_needs_named_module() -> None:
    assert mode_needs_named_module("named_shared") is True
    assert mode_needs_named_module("inline_shared") is False


def test_max_steering_configs_sizing() -> None:
    assert max_steering_configs_for("disabled", 8) == 4
    assert max_steering_configs_for("inline_shared", 8) == 4
    assert max_steering_configs_for("inline_unique", 8) == 64
    assert max_steering_configs_for("inline_unique", 40) == 80
    assert max_steering_configs_for("per_request_4", 8) == 16
    assert max_steering_configs_for("per_request_8", 8) == 32


def test_steering_config_for() -> None:
    cfg = steering_config_for("disabled", 8)
    assert isinstance(cfg, SteeringConfig)
    assert cfg.enable_steering is False
    assert cfg.max_steering_configs == 4
    cfg2 = steering_config_for("inline_unique", 8, enable_prefix_caching=False)
    assert cfg2.enable_steering is True
    assert cfg2.max_steering_configs == 64
    assert cfg2.enable_prefix_caching is False


# -- build_mode_requests -----------------------------------------------------


def test_disabled_and_idle_have_no_steering() -> None:
    for mode in ("disabled", "enabled_idle"):
        reqs = build_mode_requests(mode, batch_size=4, **KW)
        assert len(reqs) == 4
        assert all(r.steering is None for r in reqs)


def test_inline_shared_uses_one_shared_spec() -> None:
    reqs = build_mode_requests("inline_shared", batch_size=5, **KW)
    assert len(reqs) == 5
    assert all(isinstance(r.steering, SteeringSpec) for r in reqs)
    first = reqs[0].steering
    assert all(r.steering == first for r in reqs)


def test_inline_unique_uses_distinct_specs() -> None:
    reqs = build_mode_requests("inline_unique", batch_size=4, **KW)
    assert len(reqs) == 4
    specs = [r.steering for r in reqs]
    assert all(isinstance(s, SteeringSpec) for s in specs)
    # All distinct.
    seen = [s.to_vector_dict() for s in specs]  # type: ignore[union-attr]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert seen[i] != seen[j]


def test_named_shared_uses_named_module_ref() -> None:
    reqs = build_mode_requests("named_shared", batch_size=3, **KW)
    assert len(reqs) == 3
    for r in reqs:
        assert isinstance(r.steering, NamedModuleRef)
        assert r.steering.name == NAMED_MODULE
        assert r.steering.scale == 1.0


def test_named_shared_fallback_uses_inline_spec() -> None:
    reqs = build_mode_requests("named_shared", batch_size=3, named_fallback=True, **KW)
    assert len(reqs) == 3
    assert all(isinstance(r.steering, SteeringSpec) for r in reqs)
    # Fallback vectors match the inline_shared spec (same module vectors).
    inline = build_mode_requests("inline_shared", batch_size=3, **KW)
    assert reqs[0].steering == inline[0].steering


def test_per_request_cycles_distinct_specs() -> None:
    reqs = build_mode_requests("per_request_4", batch_size=8, **KW)
    assert len(reqs) == 8
    dicts = [r.steering.to_vector_dict() for r in reqs]  # type: ignore[union-attr]
    # 4 distinct specs cycled: index i uses spec i % 4.
    for i in range(8):
        assert dicts[i] == dicts[i % 4]
    # The 4 base specs are pairwise distinct.
    for i in range(4):
        for j in range(i + 1, 4):
            assert dicts[i] != dicts[j]


# -- ModeBenchmark degradation / registration --------------------------------


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        model=MODEL,
        warmup=1,
        iters=1,
        max_tokens=8,
        layer=3,
        hook="post_block",
        output_dir="/tmp",
        tag="",
    )


class _FakeEngine(SteeringEngine):
    name = "fake"
    capabilities = Capabilities(batching=True)

    def __init__(self, named: bool) -> None:
        self.capabilities = Capabilities(batching=True, named_modules=named)
        self.registered: list[tuple[str, SteeringSpec]] = []

    def load(self, model_id: str, *, steering_config=None, **opts: object) -> None:
        pass

    def register_module(self, name, spec, *, replace=True, prefill=None, decode=None):
        self.registered.append((name, spec))

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [GenerationResult(output_tokens=1) for _ in requests]

    def memory_allocated_mb(self) -> float:
        return 0.0

    def teardown(self) -> None:
        pass


def _bench(engine: SteeringEngine, mode: str) -> LatencyBenchmark:
    return LatencyBenchmark(
        engine,
        _config(),
        mode=mode,
        batch_size=2,
        prompt_len=8,
        enable_prefix_caching=True,
    )


def test_named_engine_registers_module_on_after_load() -> None:
    engine = _FakeEngine(named=True)
    bench = _bench(engine, "named_shared")
    assert bench.named_fallback() is False
    bench.after_load()
    assert len(engine.registered) == 1
    assert engine.registered[0][0] == NAMED_MODULE
    reqs = bench.build_requests()
    assert all(isinstance(r.steering, NamedModuleRef) for r in reqs)


def test_non_named_engine_degrades_named_shared(capsys) -> None:
    engine = _FakeEngine(named=False)
    bench = _bench(engine, "named_shared")
    assert bench.named_fallback() is True
    bench.after_load()  # should NOT register; prints a notice
    assert engine.registered == []
    out = capsys.readouterr().out
    assert "degraded" in out
    reqs = bench.build_requests()
    assert all(isinstance(r.steering, SteeringSpec) for r in reqs)
    params = bench.parameters()
    assert params.get("named_shared_fallback") is True


def test_mode_benchmark_steering_config_and_params() -> None:
    engine = _FakeEngine(named=True)
    bench = _bench(engine, "inline_unique")
    cfg = bench.steering_config()
    assert cfg.enable_steering is True
    assert cfg.max_steering_configs == 64  # max(64, 2*2)
    params = bench.parameters()
    assert params["mode"] == "inline_unique"
    assert params["batch_size"] == 2
    assert params["max_steering_configs"] == 64
