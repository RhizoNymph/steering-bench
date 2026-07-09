"""PatchSweepResult / PatchSweepArgmax construction, to_dict, and framing.

No GPU, no transformer_lens, no httpx.
"""

from __future__ import annotations

import dataclasses

import pytest

from steering_bench.engine.patch_sweep import (
    PatchSweepArgmax,
    PatchSweepError,
    PatchSweepRequest,
    PatchSweepResult,
)


def _tl_result() -> PatchSweepResult:
    return PatchSweepResult(
        variant="tl_batched",
        cells=140,
        n_layers=28,
        n_positions=5,
        wall_s=1.23,
        cells_per_s=113.8,
        clean_logprob=-0.5,
        corrupt_logprob=-8.0,
        argmax=PatchSweepArgmax(layer=3, position=4, recovered=0.99),
    )


def test_argmax_to_dict() -> None:
    a = PatchSweepArgmax(layer=1, position=2, recovered=0.5)
    assert a.to_dict() == {"layer": 1, "position": 2, "recovered": 0.5}


def test_result_frozen() -> None:
    r = _tl_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.cells = 1  # type: ignore[misc]


def test_tl_only_fields_default_none() -> None:
    r = _tl_result()
    assert r.noise_floor is None
    assert r.auto_captured is None
    assert r.skipped is None


def test_tl_to_dict_omits_vllm_only_fields() -> None:
    d = _tl_result().to_dict()
    assert "noise_floor" not in d
    assert "auto_captured" not in d
    assert "skipped" not in d
    assert d["variant"] == "tl_batched"
    assert d["argmax"] == {"layer": 3, "position": 4, "recovered": 0.99}
    assert d["cells"] == 140


def test_vllm_to_dict_includes_vllm_only_fields() -> None:
    r = PatchSweepResult(
        variant="vllm_sweep",
        cells=140,
        n_layers=28,
        n_positions=5,
        wall_s=2.0,
        cells_per_s=70.0,
        clean_logprob=-0.5,
        corrupt_logprob=-8.0,
        argmax=PatchSweepArgmax(layer=3, position=4, recovered=0.99),
        noise_floor=-7.5,
        auto_captured=True,
        skipped=0,
    )
    d = r.to_dict()
    assert d["noise_floor"] == -7.5
    assert d["auto_captured"] is True
    assert d["skipped"] == 0


def test_request_validation() -> None:
    req = PatchSweepRequest(clean="a", corrupt="b", answer=" c")
    assert req.variant == "batched"
    assert req.n_layers == 28
    with pytest.raises(PatchSweepError):
        PatchSweepRequest(clean="", corrupt="b", answer="c")
    with pytest.raises(PatchSweepError):
        PatchSweepRequest(clean="a", corrupt="b", answer="c", n_layers=0)
    with pytest.raises(PatchSweepError):
        PatchSweepRequest(clean="a", corrupt="b", answer="c", logits_chunk_budget=0)
