"""Tests for the ``GenerationResult.output_tokens_exact`` honesty flag."""

from __future__ import annotations

import dataclasses

import pytest

from steering_bench.engine.spec import GenerationResult, SteeringSpecError


def test_output_tokens_exact_defaults_true() -> None:
    result = GenerationResult(output_tokens=5)
    assert result.output_tokens_exact is True


def test_output_tokens_exact_can_be_false() -> None:
    result = GenerationResult(output_tokens=5, output_tokens_exact=False)
    assert result.output_tokens_exact is False


def test_generation_result_is_frozen() -> None:
    result = GenerationResult(output_tokens=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.output_tokens_exact = False  # type: ignore[misc]


def test_generation_result_round_trips() -> None:
    result = GenerationResult(output_tokens=3, text="hi", output_tokens_exact=False)
    assert dataclasses.replace(result) == result
    assert result.output_tokens == 3
    assert result.text == "hi"
    assert result.output_tokens_exact is False


def test_negative_output_tokens_rejected() -> None:
    with pytest.raises(SteeringSpecError):
        GenerationResult(output_tokens=-1)
