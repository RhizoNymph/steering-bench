"""Tests for NamedModuleRef.scale (Phase 3)."""

from __future__ import annotations

import dataclasses

import pytest

from steering_bench.engine.spec import NamedModuleRef, SteeringSpecError


def test_scale_defaults_to_one() -> None:
    ref = NamedModuleRef("mymod")
    assert ref.name == "mymod"
    assert ref.scale == 1.0


def test_scale_explicit() -> None:
    ref = NamedModuleRef("mymod", scale=2.5)
    assert ref.scale == 2.5


def test_scale_coerced_to_float() -> None:
    ref = NamedModuleRef("mymod", scale=3)
    assert isinstance(ref.scale, float)
    assert ref.scale == 3.0


def test_frozen() -> None:
    ref = NamedModuleRef("mymod")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.scale = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_scale_rejected(bad: float) -> None:
    with pytest.raises(SteeringSpecError):
        NamedModuleRef("mymod", scale=bad)


def test_empty_name_rejected() -> None:
    with pytest.raises(SteeringSpecError):
        NamedModuleRef("   ")


def test_negative_scale_allowed() -> None:
    # Negative scales (subtracting a direction) are legitimate.
    ref = NamedModuleRef("mymod", scale=-1.0)
    assert ref.scale == -1.0
