"""Serving-oriented SteeringSpec fields: per-row scales + prefill/decode split.

Runs WITHOUT a GPU and WITHOUT vllm/openai installed. Verifies the Phase-5
additions preserve all pre-serving invariants (defaults None behave as before,
frozen).
"""

from __future__ import annotations

import dataclasses

import pytest

from steering_bench.engine.spec import (
    PhaseSteeringSpec,
    SteeringSpec,
    SteeringSpecError,
)


def _multi_layer_spec(scales=None) -> SteeringSpec:
    return SteeringSpec(
        vectors={"post_block": {0: (0.1, 0.2), 1: (0.3, 0.4), 2: (0.5, 0.6)}},
        scales=scales,
    )


def test_scales_default_none_behaves_as_before() -> None:
    spec = _multi_layer_spec()
    assert spec.scales is None
    assert spec.scales_for("post_block") is None
    assert spec.num_vectors() == 3


def test_scales_length_must_match_vector_count() -> None:
    # 3 vectors -> 3 scales ok.
    spec = _multi_layer_spec(scales=(0.5, 0.55, 0.6))
    assert spec.scales == (0.5, 0.55, 0.6)
    assert all(isinstance(x, float) for x in spec.scales)
    with pytest.raises(SteeringSpecError):
        _multi_layer_spec(scales=(0.5, 0.55))  # too few
    with pytest.raises(SteeringSpecError):
        _multi_layer_spec(scales=(0.5, 0.55, 0.6, 0.65))  # too many


def test_scales_must_be_finite() -> None:
    with pytest.raises(SteeringSpecError):
        _multi_layer_spec(scales=(0.5, float("inf"), 0.6))
    with pytest.raises(SteeringSpecError):
        _multi_layer_spec(scales=(0.5, float("nan"), 0.6))


def test_scales_for_slices_per_hook_in_sorted_layer_order() -> None:
    spec = SteeringSpec(
        vectors={
            "pre_attn": {5: (1.0,), 2: (2.0,)},  # sorted -> layers 2, 5
            "post_mlp": {0: (3.0,)},
        },
        scales=(0.1, 0.2, 0.9),  # pre_attn: [0.1, 0.2], post_mlp: [0.9]
    )
    assert spec.num_vectors() == 3
    assert spec.scales_for("pre_attn") == (0.1, 0.2)
    assert spec.scales_for("post_mlp") == (0.9,)


def test_scales_for_unknown_hook_raises() -> None:
    spec = _multi_layer_spec(scales=(0.5, 0.55, 0.6))
    with pytest.raises(SteeringSpecError):
        spec.scales_for("nope")


def test_with_scales_returns_new_frozen_copy() -> None:
    spec = _multi_layer_spec()
    scaled = spec.with_scales((0.5, 0.6, 0.7))
    assert spec.scales is None  # original untouched
    assert scaled.scales == (0.5, 0.6, 0.7)
    assert scaled.to_vector_dict() == spec.to_vector_dict()
    assert scaled.with_scales(None).scales is None


def test_spec_is_frozen() -> None:
    spec = _multi_layer_spec(scales=(0.5, 0.55, 0.6))
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.scales = (1.0, 1.0, 1.0)  # type: ignore[misc]


def test_phase_steering_spec_defaults_none() -> None:
    base = _multi_layer_spec()
    phase = PhaseSteeringSpec(base=base)
    assert phase.base is base
    assert phase.prefill is None
    assert phase.decode is None


def test_phase_steering_spec_carries_variants() -> None:
    base = _multi_layer_spec()
    prefill = SteeringSpec.single("post_block", 0, [9.0, 9.0])
    decode = SteeringSpec.single("post_block", 0, [8.0, 8.0])
    phase = PhaseSteeringSpec(base=base, prefill=prefill, decode=decode)
    assert phase.prefill is prefill
    assert phase.decode is decode


def test_phase_steering_spec_validates_types() -> None:
    with pytest.raises(SteeringSpecError):
        PhaseSteeringSpec(base=object())  # type: ignore[arg-type]
    with pytest.raises(SteeringSpecError):
        PhaseSteeringSpec(base=_multi_layer_spec(), prefill="nope")  # type: ignore[arg-type]


def test_phase_steering_spec_is_frozen() -> None:
    phase = PhaseSteeringSpec(base=_multi_layer_spec())
    with pytest.raises(dataclasses.FrozenInstanceError):
        phase.decode = _multi_layer_spec()  # type: ignore[misc]
