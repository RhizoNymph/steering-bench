"""Pure-logic tests for the repeng engine adapter (no repeng/torch)."""

from __future__ import annotations

import sys

import pytest

from steering_bench.engine.base import EngineError
from steering_bench.engine.engines.repeng import (
    RepengSteeringEngine,
    control_layers_for,
    resolve_single,
    spec_cache_key,
)
from steering_bench.engine.spec import SteeringSpec
from steering_bench.vectors import random_steering_vectors


def test_import_does_not_pull_repeng() -> None:
    assert "steering_bench.engine.engines.repeng" in sys.modules


def test_capabilities_declaration() -> None:
    caps = RepengSteeringEngine.capabilities
    assert caps.batching is False  # sequential
    assert caps.multi_layer is True  # bands around the target
    assert caps.named_modules is False
    assert caps.multi_hook is False
    assert caps.capture is False


def test_control_layers_band() -> None:
    # Interior layer: [layer-2 .. layer+2].
    assert control_layers_for(8, num_layers=32) == [6, 7, 8, 9, 10]
    # Clamped at the bottom.
    assert control_layers_for(1, num_layers=32) == [0, 1, 2, 3]
    # Clamped at the top.
    assert control_layers_for(31, num_layers=32) == [29, 30, 31]


def test_resolve_single_rejects_multi_hook() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=3, hook_points=["pre_attn", "post_mlp"], seed=1
    )
    spec = SteeringSpec.from_vector_dict(raw)
    with pytest.raises(EngineError):
        resolve_single(spec)


def test_cache_key_same_spec_is_hit() -> None:
    # Two independently-built but structurally-identical specs hash equal.
    a = SteeringSpec.single("post_block", 5, [0.1, 0.2, 0.3])
    b = SteeringSpec.single("post_block", 5, [0.1, 0.2, 0.3])
    assert spec_cache_key(a) == spec_cache_key(b)
    # Keys are hashable (usable as dict keys / set members).
    assert len({spec_cache_key(a), spec_cache_key(b)}) == 1


def test_cache_key_different_vector_rebuilds() -> None:
    a = SteeringSpec.single("post_block", 5, [0.1, 0.2, 0.3])
    diff_vec = SteeringSpec.single("post_block", 5, [0.9, 0.2, 0.3])
    diff_layer = SteeringSpec.single("post_block", 6, [0.1, 0.2, 0.3])
    diff_hook = SteeringSpec.single("pre_attn", 5, [0.1, 0.2, 0.3])
    assert spec_cache_key(a) != spec_cache_key(diff_vec)
    assert spec_cache_key(a) != spec_cache_key(diff_layer)
    assert spec_cache_key(a) != spec_cache_key(diff_hook)
