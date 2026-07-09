"""Harness capture benchmark: spec builders + capability gating (no GPU/vllm)."""

from __future__ import annotations

import pytest

from steering_bench.engine.base import Capabilities
from steering_bench.harness.benchmarks.capture import (
    CAPTURE_CONFIGS,
    CaptureBenchmark,
    build_consumer_specs,
    build_request_capture,
)
from steering_bench.harness.benchmarks.registry import BENCHMARK_REGISTRY, get_benchmark

MODEL = "google/gemma-3-4b-it"  # in the static MODEL_CONFIGS table (34 layers)


def test_capture_registered() -> None:
    assert "capture" in BENCHMARK_REGISTRY
    assert get_benchmark("capture") is CaptureBenchmark


def test_capture_requires_capability() -> None:
    assert CaptureBenchmark.required_capabilities == Capabilities(capture=True)


def test_baseline_specs_empty() -> None:
    assert build_consumer_specs("baseline", model=MODEL, layer=6, hook="post_block") == []


def test_logging_minimal_spec() -> None:
    specs = build_consumer_specs("logging_minimal", model=MODEL, layer=6, hook="post_block")
    assert len(specs) == 1
    cfg = specs[0].to_llm_config()
    assert cfg["name"] == "logging"
    assert cfg["params"]["hooks"] == {"post_block": [6]}


def test_logging_max_covers_all_layers() -> None:
    specs = build_consumer_specs("logging_max", model=MODEL, layer=6, hook="post_block")
    assert specs[0].params["hooks"]["post_block"] == list(range(34))
    assert specs[0].params["positions"] == "all"


def test_logging_3x_distinct_instance_names() -> None:
    specs = build_consumer_specs("logging_3x", model=MODEL, layer=6, hook="post_block")
    assert [s.instance_name for s in specs] == ["log_a", "log_b", "log_c"]


def test_layer_clamped_to_model() -> None:
    # layer 999 clamps to num_layers - 1 (33 for gemma-3-4b-it).
    specs = build_consumer_specs("logging_minimal", model=MODEL, layer=999, hook="post_block")
    assert specs[0].params["hooks"]["post_block"] == [33]


def test_filesystem_needs_root() -> None:
    with pytest.raises(ValueError):
        build_consumer_specs("filesystem_minimal", model=MODEL, layer=6, hook="post_block")


def test_filesystem_request_capture() -> None:
    rc = build_request_capture("filesystem_minimal", model=MODEL, layer=6, hook="post_block")
    assert rc is not None
    field = rc.to_capture_field()["filesystem"]
    assert field["hooks"] == {"post_block": [6]}
    assert field["tag"] == "benchmark"


def test_non_filesystem_no_request_capture() -> None:
    for cfg in CAPTURE_CONFIGS:
        if cfg == "filesystem_minimal":
            continue
        assert build_request_capture(cfg, model=MODEL, layer=6, hook="post_block") is None


def test_driver_needs_instance() -> None:
    with pytest.raises(ValueError):
        build_consumer_specs("driver_minimal", model=MODEL, layer=6, hook="post_block")


def test_driver_instance_passthrough() -> None:
    sentinel = object()
    specs = build_consumer_specs(
        "driver_minimal", model=MODEL, layer=6, hook="post_block", driver_instance=sentinel
    )
    assert specs[0].location == "driver"
    assert specs[0].to_llm_config() is sentinel
