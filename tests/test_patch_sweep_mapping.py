"""Pure dict -> PatchSweepResult mapping for both backend dict shapes.

Feeds canned dicts matching the documented outputs of
``external/tl_patching.py`` and ``external/vllm_patch_sweep.py`` and asserts the
field mapping + argmax reduction. No transformer_lens / httpx import.
"""

from __future__ import annotations

from steering_bench.engine.patch_sweep import PatchSweepResult

# The exact dict shape ``tl_patching.run_patch_sweep`` returns.
TL_DICT = {
    "variant": "tl_batched",
    "cells": 140,
    "n_layers": 28,
    "n_positions": 5,
    "wall_s": 1.234,
    "cells_per_s": 113.5,
    "clean_logprob": -0.4021,
    "corrupt_logprob": -9.1234,
    "argmax": {"layer": 7, "position": 4, "recovered": 0.9876},
}

# The exact dict shape ``vllm_patch_sweep.run_patch_sweep`` returns.
VLLM_DICT = {
    "variant": "vllm_sweep",
    "cells": 138,
    "n_layers": 28,
    "n_positions": 5,
    "wall_s": 2.5,
    "cells_per_s": 55.2,
    "clean_logprob": -0.40,
    "corrupt_logprob": -9.12,
    "noise_floor": -8.9,
    "auto_captured": True,
    "skipped": 2,
    "argmax": {"layer": 7, "position": 4, "recovered": 0.9876},
}


def test_from_tl_dict() -> None:
    r = PatchSweepResult.from_tl_dict(TL_DICT)
    assert r.variant == "tl_batched"
    assert r.cells == 140
    assert r.n_layers == 28
    assert r.n_positions == 5
    assert r.wall_s == 1.234
    assert r.cells_per_s == 113.5
    assert r.clean_logprob == -0.4021
    assert r.corrupt_logprob == -9.1234
    # argmax reduction
    assert r.argmax.layer == 7
    assert r.argmax.position == 4
    assert r.argmax.recovered == 0.9876
    # vLLM-only fields absent for tl
    assert r.noise_floor is None
    assert r.auto_captured is None
    assert r.skipped is None


def test_tl_roundtrip_reproduces_source_dict() -> None:
    # to_dict of a from_tl_dict result reproduces the original tl dict exactly.
    assert PatchSweepResult.from_tl_dict(TL_DICT).to_dict() == TL_DICT


def test_from_vllm_dict() -> None:
    r = PatchSweepResult.from_vllm_dict(VLLM_DICT)
    assert r.variant == "vllm_sweep"
    assert r.cells == 138
    assert r.argmax.layer == 7
    assert r.argmax.position == 4
    assert r.argmax.recovered == 0.9876
    assert r.noise_floor == -8.9
    assert r.auto_captured is True
    assert r.skipped == 2


def test_vllm_roundtrip_reproduces_source_dict() -> None:
    assert PatchSweepResult.from_vllm_dict(VLLM_DICT).to_dict() == VLLM_DICT


def test_vllm_dict_with_null_argmax() -> None:
    # An empty vLLM grid can produce a null argmax; the mapper tolerates it.
    d = {**VLLM_DICT, "argmax": {"layer": None, "position": None, "recovered": None}}
    r = PatchSweepResult.from_vllm_dict(d)
    assert r.argmax.layer is None
    assert r.argmax.position is None
    assert r.argmax.recovered is None
