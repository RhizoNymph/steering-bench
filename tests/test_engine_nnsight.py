"""Pure-logic tests for the nnsight engine adapter (no nnsight/torch)."""

from __future__ import annotations

import sys

import pytest

from steering_bench.engine.base import EngineError
from steering_bench.engine.engines.nnsight import (
    NnsightSteeringEngine,
    batch_placeholder_results,
    layer_path,
    resolve_single,
)
from steering_bench.engine.spec import GenerationRequest, SteeringSpec
from steering_bench.vectors import random_steering_vectors


def test_import_does_not_pull_nnsight() -> None:
    assert "steering_bench.engine.engines.nnsight" in sys.modules


def test_capabilities_declaration() -> None:
    caps = NnsightSteeringEngine.capabilities
    # Pseudo-batch, everything else off.
    assert caps.batching is True
    assert caps.named_modules is False
    assert caps.multi_layer is False
    assert caps.multi_hook is False
    assert caps.capture is False


def test_resolve_single_unpacks() -> None:
    spec = SteeringSpec.single("post_block", 7, [0.1, 0.2, 0.3])
    hook, layer, vec = resolve_single(spec)
    assert (hook, layer) == ("post_block", 7)
    assert vec == (0.1, 0.2, 0.3)


def test_layer_path_mapping() -> None:
    assert layer_path(7) == "model.layers.7.output[0]"


def test_resolve_single_rejects_multi_hook() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=3, hook_points=["pre_attn", "post_mlp"], seed=1
    )
    spec = SteeringSpec.from_vector_dict(raw)
    with pytest.raises(EngineError):
        resolve_single(spec)


def test_resolve_single_rejects_multi_layer() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=3, hook_points=["post_block"], seed=2
    )
    spec = SteeringSpec.from_vector_dict(raw)  # 3 layers for the hook
    with pytest.raises(EngineError):
        resolve_single(spec)


def test_batch_path_marks_output_tokens_inexact() -> None:
    spec = SteeringSpec.single("post_block", 3, [0.1, 0.2])
    requests = [
        GenerationRequest(prompt="a", max_tokens=16, steering=spec),
        GenerationRequest(prompt="b", max_tokens=16, steering=spec),
    ]
    results = batch_placeholder_results(requests)
    assert len(results) == 2
    assert all(r.output_tokens_exact is False for r in results)
    assert all(r.output_tokens == 16 for r in results)
