"""Translation tests for the vLLM engine adapter.

`spec_to_native` and `named_ref_to_kwargs` are pure module-level
functions that must NOT require vllm to be importable.
"""

from __future__ import annotations

import sys

import pytest

from steering_bench.engine.engines.vllm import (
    VllmSteeringEngine,
    device_used_memory_mb,
    named_payload_from_spec,
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


def test_named_ref_to_kwargs_emits_name_scale_tuple() -> None:
    # The vLLM fork expects a (name, scale) TUPLE, not a bare name string.
    assert named_ref_to_kwargs(NamedModuleRef("mymod")) == {
        "steering_module_ref": ("mymod", 1.0)
    }
    assert named_ref_to_kwargs(NamedModuleRef("mymod", scale=2.5)) == {
        "steering_module_ref": ("mymod", 2.5)
    }


def test_steering_kwargs_variants() -> None:
    assert steering_kwargs(None) == {}
    spec = SteeringSpec.single("post_block", 1, [1.0])
    assert steering_kwargs(spec) == {"steering_vectors": {"post_block": {1: [1.0]}}}
    assert steering_kwargs(NamedModuleRef("m")) == {
        "steering_module_ref": ("m", 1.0)
    }
    assert steering_kwargs(NamedModuleRef("m", scale=0.5)) == {
        "steering_module_ref": ("m", 0.5)
    }


def test_named_payload_from_spec_shape() -> None:
    spec = SteeringSpec.single("post_block", 3, [0.1, 0.2])
    payload = named_payload_from_spec(spec)
    assert payload == {"vectors": {"post_block": {3: [0.1, 0.2]}}}


def test_named_payload_from_spec_multi_hook_multi_layer() -> None:
    raw = random_steering_vectors(
        hidden_size=4, num_layers=3, hook_points=["pre_attn", "post_mlp"], seed=7
    )
    spec = SteeringSpec.from_vector_dict(raw)
    payload = named_payload_from_spec(spec)
    assert set(payload) == {"vectors"}
    vecs = payload["vectors"]
    assert set(vecs) == {"pre_attn", "post_mlp"}
    assert set(vecs["pre_attn"]) == {0, 1, 2}
    # Inner values are plain lists of floats (JSON / RPC friendly).
    assert isinstance(vecs["pre_attn"][0], list)
    assert all(isinstance(x, float) for x in vecs["pre_attn"][0])
    assert vecs == raw


def test_named_payload_coerces_numpy_without_importing_vllm() -> None:
    np = pytest.importorskip("numpy")
    # A spec whose vectors originate from numpy arrays still yields plain lists.
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    spec = SteeringSpec.single("post_block", 0, arr)
    payload = named_payload_from_spec(spec)
    inner = payload["vectors"]["post_block"][0]
    assert inner == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in inner)


def test_named_payload_prefill_decode_split() -> None:
    spec = SteeringSpec.single("post_block", 1, [1.0, 2.0])
    prefill = SteeringSpec.single("post_block", 1, [3.0, 4.0])
    decode = SteeringSpec.single("post_block", 1, [5.0, 6.0])
    payload = named_payload_from_spec(spec, prefill=prefill, decode=decode)
    assert payload["vectors"] == {"post_block": {1: [1.0, 2.0]}}
    assert payload["prefill_vectors"] == {"post_block": {1: [3.0, 4.0]}}
    assert payload["decode_vectors"] == {"post_block": {1: [5.0, 6.0]}}


def test_steering_kwargs_rejects_offline_per_row_scales() -> None:
    """Finding #2: offline inline steering has no per-row scale field, so a
    SteeringSpec carrying scales must raise rather than silently drop them."""
    import dataclasses

    spec = SteeringSpec.single("post_block", 0, [1.0, 2.0])
    scaled = dataclasses.replace(spec, scales=(0.5,))
    with pytest.raises(ValueError, match="per-row scales"):
        steering_kwargs(scaled)
    # an unscaled spec is unaffected
    assert "steering_vectors" in steering_kwargs(spec)


def test_device_used_memory_mb_computes_total_minus_free() -> None:
    """Finding #1: the vLLM engine reports DEVICE-wide used memory
    (total - free) in MB, since its model lives in an EngineCore subprocess and
    driver-process ``memory_allocated()`` reads 0. Inject a (free, total)
    reporter to test the pure computation without a real GPU."""
    free = 3 * 1024 * 1024 * 1024  # 3 GiB free
    total = 8 * 1024 * 1024 * 1024  # 8 GiB total
    used_mb = device_used_memory_mb(lambda: (free, total))
    assert isinstance(used_mb, float)
    assert used_mb == (total - free) / (1024 * 1024)  # 5 GiB in MB == 5120.0
    assert used_mb == 5120.0


def test_memory_allocated_mb_returns_float() -> None:
    # On a CPU-only host CUDA is unavailable, so the honest fallback is 0.0 —
    # but always a float, never an exception.
    mb = VllmSteeringEngine().memory_allocated_mb()
    assert isinstance(mb, float)
