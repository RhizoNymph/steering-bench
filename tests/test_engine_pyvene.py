"""Pure-logic tests for the pyvene engine adapter (no pyvene/torch)."""

from __future__ import annotations

import sys

import pytest

from steering_bench.engine.base import EngineError
from steering_bench.engine.engines.pyvene import (
    PyveneSteeringEngine,
    component_for,
    resolve_single,
    spec_cache_key,
)
from steering_bench.engine.spec import SteeringSpec
from steering_bench.vectors import random_steering_vectors


def test_import_does_not_pull_pyvene() -> None:
    assert "steering_bench.engine.engines.pyvene" in sys.modules


def test_capabilities_declaration() -> None:
    caps = PyveneSteeringEngine.capabilities
    assert caps.batching is False
    assert caps.named_modules is False
    assert caps.multi_layer is False
    assert caps.multi_hook is False
    assert caps.capture is False


def test_component_map() -> None:
    assert component_for("pre_attn") == "block_input"
    assert component_for("post_attn") == "block_output"
    assert component_for("post_mlp") == "block_output"
    # Unknown hooks fall back to block_output.
    assert component_for("post_block") == "block_output"


def test_resolve_single_rejects_multi_layer() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=3, hook_points=["post_block"], seed=3
    )
    spec = SteeringSpec.from_vector_dict(raw)
    with pytest.raises(EngineError):
        resolve_single(spec)


def test_cache_key_same_spec_is_hit() -> None:
    a = SteeringSpec.single("post_mlp", 4, [0.5, 0.6])
    b = SteeringSpec.single("post_mlp", 4, [0.5, 0.6])
    assert spec_cache_key(a) == spec_cache_key(b)


def test_cache_key_different_spec_rebuilds() -> None:
    a = SteeringSpec.single("post_mlp", 4, [0.5, 0.6])
    assert spec_cache_key(a) != spec_cache_key(
        SteeringSpec.single("post_mlp", 4, [0.5, 0.7])
    )
    assert spec_cache_key(a) != spec_cache_key(
        SteeringSpec.single("pre_attn", 4, [0.5, 0.6])
    )
