"""Typed domain model tests for the engine seam.

These run WITHOUT a GPU and WITHOUT vllm / transformer_lens installed.
"""

from __future__ import annotations

import pytest

from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    SteeringSpec,
    SteeringSpecError,
)
from steering_bench.vectors import random_steering_vectors


def test_from_vector_dict_single_hook() -> None:
    raw = random_steering_vectors(
        hidden_size=8,
        num_layers=4,
        hook_points=["post_block"],
        seed=1,
    )
    spec = SteeringSpec.from_vector_dict(raw)

    assert spec.hooks() == ("post_block",)
    assert spec.layers("post_block") == (0, 1, 2, 3)
    assert spec.dim("post_block") == 8
    # Inner vectors are normalized to tuples of floats.
    assert isinstance(spec.vectors["post_block"][0], tuple)
    assert all(isinstance(x, float) for x in spec.vectors["post_block"][0])


def test_from_vector_dict_multi_hook_multi_layer() -> None:
    raw = random_steering_vectors(
        hidden_size=6,
        num_layers=3,
        hook_points=["pre_attn", "post_mlp"],
        seed=7,
    )
    spec = SteeringSpec.from_vector_dict(raw)

    assert set(spec.hooks()) == {"pre_attn", "post_mlp"}
    assert spec.layers("pre_attn") == (0, 1, 2)
    assert spec.dim("post_mlp") == 6
    assert spec.is_multi_hook()
    assert spec.is_multi_layer()
    assert not spec.is_single_hook_single_layer()


def test_single_helper() -> None:
    spec = SteeringSpec.single("post_block", 8, [0.1, 0.2, 0.3])
    assert spec.hooks() == ("post_block",)
    assert spec.layers("post_block") == (8,)
    assert spec.dim("post_block") == 3
    assert spec.is_single_hook_single_layer()
    assert not spec.is_multi_hook()
    assert not spec.is_multi_layer()


def test_layer_subset_introspection() -> None:
    raw = random_steering_vectors(
        hidden_size=5,
        num_layers=10,
        hook_points=["post_block"],
        layer_subset=[2, 5, 9],
        seed=3,
    )
    spec = SteeringSpec.from_vector_dict(raw)
    assert spec.layers("post_block") == (2, 5, 9)
    assert spec.is_multi_layer()


def test_to_vector_dict_roundtrip() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=2, hook_points=["post_block"], seed=11
    )
    spec = SteeringSpec.from_vector_dict(raw)
    back = spec.to_vector_dict()
    assert back == raw


def test_empty_vectors_raises() -> None:
    with pytest.raises(ValueError):
        SteeringSpec(vectors={})


def test_empty_layers_raises() -> None:
    with pytest.raises(ValueError):
        SteeringSpec(vectors={"post_block": {}})


def test_ragged_vector_lengths_raise() -> None:
    with pytest.raises(SteeringSpecError):
        SteeringSpec(vectors={"post_block": {0: (0.1, 0.2), 1: (0.3,)}})


def test_zero_length_vector_raises() -> None:
    with pytest.raises(ValueError):
        SteeringSpec(vectors={"post_block": {0: ()}})


def test_spec_error_is_value_error() -> None:
    assert issubclass(SteeringSpecError, ValueError)


def test_named_module_ref_validation() -> None:
    ref = NamedModuleRef(name="my_module")
    assert ref.name == "my_module"
    with pytest.raises(ValueError):
        NamedModuleRef(name="")
    with pytest.raises(ValueError):
        NamedModuleRef(name="   ")


def test_generation_request_with_spec() -> None:
    spec = SteeringSpec.single("post_block", 0, [1.0, 2.0])
    req = GenerationRequest(prompt="hello", max_tokens=16, steering=spec)
    assert req.steering is spec
    assert req.max_tokens == 16


def test_generation_request_with_named_ref() -> None:
    ref = NamedModuleRef(name="mod")
    req = GenerationRequest(prompt="hello", max_tokens=8, steering=ref)
    assert req.steering is ref


def test_generation_request_without_steering() -> None:
    req = GenerationRequest(prompt="hello", max_tokens=8)
    assert req.steering is None


def test_generation_request_bad_max_tokens() -> None:
    with pytest.raises(ValueError):
        GenerationRequest(prompt="hello", max_tokens=0)
    with pytest.raises(ValueError):
        GenerationRequest(prompt="hello", max_tokens=-1)


def test_generation_result() -> None:
    res = GenerationResult(output_tokens=12)
    assert res.output_tokens == 12
    assert res.text is None
    with pytest.raises(ValueError):
        GenerationResult(output_tokens=-1)
