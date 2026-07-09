"""The pure adapter-owned packer round-trips. NO vllm/openai/httpx import."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from steering_bench.engine.serving import (
    named_register_payload,
    pack_steering_vectors,
    steering_extra_body,
)
from steering_bench.engine.spec import NamedModuleRef, SteeringSpec


def _decode(entry: dict) -> np.ndarray:
    raw = base64.b64decode(entry["data"])
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(entry["shape"])


def test_pack_single_hook_multi_layer_round_trips() -> None:
    spec = SteeringSpec(
        vectors={"post_block": {2: (0.1, 0.2, 0.3), 0: (1.0, 2.0, 3.0), 1: (4.0, 5.0, 6.0)}}
    )
    packed = pack_steering_vectors(spec)
    assert set(packed) == {"post_block"}
    entry = packed["post_block"]
    assert entry["dtype"] == "float32"
    # Rows are in ASCENDING layer order.
    assert entry["layer_indices"] == [0, 1, 2]
    assert entry["shape"] == [3, 3]
    assert "scales" not in entry
    arr = _decode(entry)
    np.testing.assert_allclose(arr[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(arr[1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(arr[2], [0.1, 0.2, 0.3], rtol=1e-6)


def test_pack_with_scales_attaches_per_row_list() -> None:
    spec = SteeringSpec(
        vectors={"post_block": {0: (1.0, 1.0), 1: (2.0, 2.0)}},
        scales=(0.5, 0.75),
    )
    entry = pack_steering_vectors(spec)["post_block"]
    assert entry["scales"] == [0.5, 0.75]
    assert entry["shape"] == [2, 2]


def test_pack_multi_hook() -> None:
    spec = SteeringSpec(
        vectors={"pre_attn": {0: (1.0,), 1: (2.0,)}, "post_mlp": {0: (3.0,)}},
        scales=(0.1, 0.2, 0.3),
    )
    packed = pack_steering_vectors(spec)
    assert set(packed) == {"pre_attn", "post_mlp"}
    assert packed["pre_attn"]["scales"] == [0.1, 0.2]
    assert packed["post_mlp"]["scales"] == [0.3]
    np.testing.assert_allclose(_decode(packed["pre_attn"]), [[1.0], [2.0]])


def test_steering_extra_body_variants() -> None:
    assert steering_extra_body(None) is None
    spec = SteeringSpec.single("post_block", 0, [1.0, 2.0])
    body = steering_extra_body(spec)
    assert set(body) == {"steering_vectors"}
    assert "post_block" in body["steering_vectors"]
    # A named ref encodes to a bare name; a non-default scale is rejected
    # (covered by test_steering_extra_body_rejects_named_ref_scale).
    named = steering_extra_body(NamedModuleRef("mod"))
    assert named == {"steering_name": "mod"}


def test_packer_pulls_in_no_http_stack() -> None:
    # The adapter-owned encoders must stay pure: exercising them must not import
    # the HTTP / backend stack (those stay lazy in the concrete adapter).
    import sys

    before = set(sys.modules)
    pack_steering_vectors(SteeringSpec.single("post_block", 0, [1.0]))
    named_register_payload("m", SteeringSpec.single("post_block", 0, [1.0]))
    steering_extra_body(NamedModuleRef("m"))
    newly = set(sys.modules) - before
    assert not ({"vllm", "openai", "httpx"} & newly)


def test_steering_extra_body_rejects_named_ref_scale() -> None:
    """Finding #1: the HTTP serving path can't express a named-ref scale, so a
    non-default scale must raise rather than be silently applied as 1.0."""
    with pytest.raises(ValueError, match="cannot carry a scale"):
        steering_extra_body(NamedModuleRef("m", scale=2.0))
    # default scale (1.0) is the encodable case
    assert steering_extra_body(NamedModuleRef("m")) == {"steering_name": "m"}
