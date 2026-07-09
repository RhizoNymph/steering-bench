"""The adapter-owned register payload builder. No vllm/openai/httpx import."""

from __future__ import annotations

from steering_bench.engine.serving import named_register_payload
from steering_bench.engine.spec import PhaseSteeringSpec, SteeringSpec


def test_payload_without_prefill_decode() -> None:
    spec = SteeringSpec(vectors={"post_block": {0: (1.0, 2.0), 1: (3.0, 4.0)}})
    payload = named_register_payload("bench_named_shared", spec)
    assert payload["name"] == "bench_named_shared"
    # Register endpoint takes RAW vectors (not the packed base64 form).
    assert payload["vectors"] == {"post_block": {0: [1.0, 2.0], 1: [3.0, 4.0]}}
    assert payload["prefill_vectors"] is None
    assert payload["decode_vectors"] is None


def test_payload_with_prefill_decode_kwargs() -> None:
    base = SteeringSpec.single("post_block", 0, [1.0, 1.0])
    prefill = SteeringSpec.single("post_block", 0, [2.0, 2.0])
    decode = SteeringSpec.single("post_block", 0, [3.0, 3.0])
    payload = named_register_payload("m", base, prefill=prefill, decode=decode)
    assert payload["vectors"] == {"post_block": {0: [1.0, 1.0]}}
    assert payload["prefill_vectors"] == {"post_block": {0: [2.0, 2.0]}}
    assert payload["decode_vectors"] == {"post_block": {0: [3.0, 3.0]}}


def test_payload_from_phase_steering_spec() -> None:
    base = SteeringSpec.single("post_block", 0, [1.0])
    prefill = SteeringSpec.single("post_block", 0, [2.0])
    phase = PhaseSteeringSpec(base=base, prefill=prefill)
    payload = named_register_payload("m", phase)
    assert payload["vectors"] == {"post_block": {0: [1.0]}}
    assert payload["prefill_vectors"] == {"post_block": {0: [2.0]}}
    assert payload["decode_vectors"] is None


def test_explicit_kwargs_override_phase_fields() -> None:
    base = SteeringSpec.single("post_block", 0, [1.0])
    phase = PhaseSteeringSpec(
        base=base, decode=SteeringSpec.single("post_block", 0, [5.0])
    )
    override = SteeringSpec.single("post_block", 0, [9.0])
    payload = named_register_payload("m", phase, decode=override)
    assert payload["decode_vectors"] == {"post_block": {0: [9.0]}}
