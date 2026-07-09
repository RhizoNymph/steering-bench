"""Serving benchmark harness: registry, mode validation, request building.

No GPU, no live server, no vllm/openai import.
"""

from __future__ import annotations

import pytest

from steering_bench.engine.spec import NamedModuleRef, SteeringSpec
from steering_bench.harness.benchmarks.registry import (
    SERVING_REGISTRY,
    get_serving_benchmark,
)
from steering_bench.harness.benchmarks.serving import (
    SERVING_MODES,
    ServingBenchmark,
    ServingModeError,
    build_serving_requests,
    distinct_configs_for_mode,
    per_request_count,
    validate_modes,
)

MODEL = "google/gemma-3-4b-it"


def test_serving_registered_separately() -> None:
    assert "serving" in SERVING_REGISTRY
    assert get_serving_benchmark("serving") is ServingBenchmark
    # Serving is NOT a sync Benchmark subclass.
    from steering_bench.harness.benchmark import Benchmark

    assert not issubclass(ServingBenchmark, Benchmark)


def test_per_request_count() -> None:
    assert per_request_count("per_request_n4") == 4
    assert per_request_count("per_request_n16") == 16
    assert per_request_count("disabled") is None
    with pytest.raises(ServingModeError):
        per_request_count("per_request_nx")
    with pytest.raises(ServingModeError):
        per_request_count("per_request_n0")


def test_validate_modes() -> None:
    validate_modes(list(SERVING_MODES))
    validate_modes(["per_request_n8"])
    with pytest.raises(ServingModeError):
        validate_modes(["bogus"])


def test_distinct_configs_for_mode() -> None:
    diverse = [SteeringSpec.single("post_block", 0, [1.0]) for _ in range(4)]
    assert distinct_configs_for_mode("disabled", []) == 0
    assert distinct_configs_for_mode("enabled_idle", []) == 0
    assert distinct_configs_for_mode("named_shared", diverse) == 1
    assert distinct_configs_for_mode("all_steered_shared", diverse) == 1
    assert distinct_configs_for_mode("per_request_n4", diverse) == 4


def _prompts(n: int) -> list[str]:
    return [f"p{i}" for i in range(n)]


def test_build_requests_disabled_and_idle() -> None:
    shared = SteeringSpec.single("post_block", 0, [1.0])
    for mode in ("disabled", "enabled_idle"):
        reqs = build_serving_requests(mode, _prompts(3), 16, shared=shared, diverse=[])
        assert len(reqs) == 3
        assert all(r.steering is None for r in reqs)
        assert all(r.max_tokens == 16 for r in reqs)


def test_build_requests_all_steered_shared() -> None:
    shared = SteeringSpec.single("post_block", 0, [1.0])
    reqs = build_serving_requests(
        "all_steered_shared", _prompts(3), 8, shared=shared, diverse=[]
    )
    assert all(r.steering is shared for r in reqs)


def test_build_requests_named_shared() -> None:
    shared = SteeringSpec.single("post_block", 0, [1.0])
    reqs = build_serving_requests(
        "named_shared", _prompts(2), 8, shared=shared, diverse=[]
    )
    assert all(isinstance(r.steering, NamedModuleRef) for r in reqs)
    assert reqs[0].steering.name == "bench_named_shared"


def test_build_requests_per_request_cycles() -> None:
    shared = SteeringSpec.single("post_block", 0, [1.0])
    diverse = [SteeringSpec.single("post_block", 0, [float(i)]) for i in range(4)]
    reqs = build_serving_requests(
        "per_request_n4", _prompts(10), 8, shared=shared, diverse=diverse
    )
    assert [r.steering for r in reqs] == [diverse[i % 4] for i in range(10)]


def test_shared_spec_with_scales_matches_layer_count() -> None:
    from steering_bench.harness.benchmarks.serving import shared_spec_for

    spec = shared_spec_for(MODEL, "post_block", with_scales=True)
    # gemma-3-4b-it has 34 layers -> 34 vectors -> 34 scales.
    assert spec.num_vectors() == 34
    assert spec.scales is not None
    assert len(spec.scales) == 34
    # Deterministic ramp: scales[i] == round(0.5 + 0.05 * i, 4).
    assert spec.scales[0] == 0.5
    assert spec.scales[1] == 0.55
