"""Translation tests for the vLLM engine adapter.

`spec_to_native` and `named_ref_to_kwargs` are pure module-level
functions that must NOT require vllm to be importable.
"""

from __future__ import annotations

import sys

from steering_bench.engine.engines.vllm import (
    named_ref_to_kwargs,
    spec_to_native,
    steering_kwargs,
)
from steering_bench.engine.spec import NamedModuleRef, SteeringSpec
from steering_bench.vectors import random_steering_vectors


def test_spec_to_native_does_not_import_vllm() -> None:
    # Importing the adapter module must not have pulled in vllm.
    assert "steering_bench.engine.engines.vllm" in sys.modules
    # The translate helper is callable regardless of vllm availability.
    spec = SteeringSpec.single("post_block", 3, [0.1, 0.2])
    native = spec_to_native(spec)
    assert native == {"post_block": {3: [0.1, 0.2]}}


def test_spec_to_native_multi_hook_multi_layer() -> None:
    raw = random_steering_vectors(
        hidden_size=4,
        num_layers=3,
        hook_points=["pre_attn", "post_mlp"],
        seed=5,
    )
    spec = SteeringSpec.from_vector_dict(raw)
    native = spec_to_native(spec)

    # Structure matches the vLLM native {hook: {layer: [floats]}} format.
    assert set(native) == {"pre_attn", "post_mlp"}
    assert set(native["pre_attn"]) == {0, 1, 2}
    assert native == raw
    # Inner values are plain lists (JSON / SamplingParams friendly).
    assert isinstance(native["pre_attn"][0], list)


def test_spec_to_native_roundtrips_through_spec() -> None:
    raw = random_steering_vectors(
        hidden_size=3, num_layers=2, hook_points=["post_block"], seed=9
    )
    spec = SteeringSpec.from_vector_dict(raw)
    assert SteeringSpec.from_vector_dict(spec_to_native(spec)).to_vector_dict() == raw


def test_named_ref_to_kwargs() -> None:
    assert named_ref_to_kwargs(NamedModuleRef("mymod")) == {
        "steering_module_ref": "mymod"
    }


def test_steering_kwargs_variants() -> None:
    assert steering_kwargs(None) == {}
    spec = SteeringSpec.single("post_block", 1, [1.0])
    assert steering_kwargs(spec) == {"steering_vectors": {"post_block": {1: [1.0]}}}
    assert steering_kwargs(NamedModuleRef("m")) == {"steering_module_ref": "m"}
