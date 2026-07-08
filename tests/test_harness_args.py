"""Shared-argument tests for the harness CLI surface."""

from __future__ import annotations

import argparse

import pytest

from steering_bench.engine.registry import ENGINE_REGISTRY
from steering_bench.harness.args import add_common_args


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p


def test_defaults() -> None:
    args = _parser().parse_args([])
    assert args.warmup == 5
    assert args.iters == 20
    assert args.max_tokens == 128
    assert args.layer == 8
    assert args.hook == "post_block"
    assert args.engine == "vllm"
    assert args.tag == ""
    assert args.model  # a non-empty default model id
    assert args.output_dir  # a non-empty default output dir


def test_engine_choices_reflect_registry() -> None:
    engine_action = next(
        a for a in _parser()._actions if a.dest == "engine"
    )
    registry_names = {e.name for e in ENGINE_REGISTRY}
    assert set(engine_action.choices) == registry_names


def test_valid_engine_parses() -> None:
    for entry in ENGINE_REGISTRY:
        args = _parser().parse_args(["--engine", entry.name])
        assert args.engine == entry.name


def test_invalid_engine_rejected() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--engine", "does-not-exist"])


def test_overrides_applied() -> None:
    args = _parser().parse_args(
        [
            "--model", "meta-llama/Llama-3.2-1B",
            "--warmup", "1",
            "--iters", "3",
            "--max-tokens", "16",
            "--layer", "4",
            "--hook", "post_attn",
            "--tag", "run-x",
            "--output-dir", "/tmp/out",
        ]
    )
    assert args.model == "meta-llama/Llama-3.2-1B"
    assert args.warmup == 1
    assert args.iters == 3
    assert args.max_tokens == 16
    assert args.layer == 4
    assert args.hook == "post_attn"
    assert args.tag == "run-x"
    assert args.output_dir == "/tmp/out"


def test_composable_with_extra_args() -> None:
    p = _parser()
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args(["--batch-size", "8"])
    assert args.batch_size == 8
    assert args.warmup == 5  # common args still present
