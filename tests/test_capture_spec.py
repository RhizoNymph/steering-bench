"""CaptureConsumerSpec / RequestCapture construction + GenerationRequest opt-in.

Runs WITHOUT a GPU and WITHOUT vllm: all types are pure data.
"""

from __future__ import annotations

import pytest

from steering_bench.engine.capture import (
    CaptureConsumerSpec,
    CaptureSpecError,
    capture_consumers_arg,
)
from steering_bench.engine.spec import (
    GenerationRequest,
    RequestCapture,
    SteeringSpecError,
)


# -- CaptureConsumerSpec ------------------------------------------------------


def test_consumer_spec_defaults() -> None:
    spec = CaptureConsumerSpec(name="logging")
    assert spec.location == "worker"
    assert spec.execution == "async"
    assert spec.params == {}
    assert spec.instance is None
    assert spec.to_llm_config() == {"name": "logging", "params": {}}


def test_consumer_spec_config_dict_with_instance_name() -> None:
    spec = CaptureConsumerSpec(
        name="logging",
        instance_name="log_a",
        params={"hooks": {"post_block": [6]}, "positions": "last_prompt"},
    )
    assert spec.to_llm_config() == {
        "name": "logging",
        "params": {"hooks": {"post_block": [6]}, "positions": "last_prompt"},
        "instance_name": "log_a",
    }


def test_consumer_spec_instance_passthrough() -> None:
    sentinel = object()
    spec = CaptureConsumerSpec(name="driver_recording", location="driver", instance=sentinel)
    # A pre-built instance is passed to LLM verbatim.
    assert spec.to_llm_config() is sentinel


def test_consumer_spec_rejects_empty_name() -> None:
    with pytest.raises(CaptureSpecError):
        CaptureConsumerSpec(name="  ")


def test_consumer_spec_rejects_bad_location() -> None:
    with pytest.raises(CaptureSpecError):
        CaptureConsumerSpec(name="logging", location="gpu")  # type: ignore[arg-type]


def test_consumer_spec_rejects_bad_execution() -> None:
    with pytest.raises(CaptureSpecError):
        CaptureConsumerSpec(name="logging", execution="blocking")  # type: ignore[arg-type]


def test_consumer_spec_is_frozen() -> None:
    spec = CaptureConsumerSpec(name="logging")
    with pytest.raises(Exception):
        spec.name = "other"  # type: ignore[misc]


def test_capture_consumers_arg_empty_is_none() -> None:
    assert capture_consumers_arg([]) is None
    assert capture_consumers_arg(None) is None


def test_capture_consumers_arg_maps_specs() -> None:
    specs = [
        CaptureConsumerSpec(name="logging", params={"level": "WARNING"}),
        CaptureConsumerSpec(name="filesystem", params={"root": "/tmp/x"}),
    ]
    arg = capture_consumers_arg(specs)
    assert arg == [
        {"name": "logging", "params": {"level": "WARNING"}},
        {"name": "filesystem", "params": {"root": "/tmp/x"}},
    ]


# -- RequestCapture -----------------------------------------------------------


def test_request_capture_to_field() -> None:
    cap = RequestCapture(consumer="filesystem", hooks={"post_block": [6]})
    assert cap.to_capture_field() == {
        "filesystem": {
            "request_id": "bench",
            "tag": "bench",
            "hooks": {"post_block": [6]},
            "positions": "last_prompt",
        }
    }


def test_request_capture_custom_fields() -> None:
    cap = RequestCapture(
        consumer="filesystem",
        hooks={"post_block": [1, 2], "post_attn": [3]},
        positions="all",
        request_id="r7",
        tag="t7",
    )
    field = cap.to_capture_field()["filesystem"]
    assert field["positions"] == "all"
    assert field["request_id"] == "r7"
    assert field["tag"] == "t7"
    assert field["hooks"] == {"post_block": [1, 2], "post_attn": [3]}


def test_request_capture_rejects_empty_consumer() -> None:
    with pytest.raises(SteeringSpecError):
        RequestCapture(consumer="", hooks={"post_block": [0]})


def test_request_capture_rejects_empty_hooks() -> None:
    with pytest.raises(SteeringSpecError):
        RequestCapture(consumer="filesystem", hooks={})


def test_request_capture_coerces_layer_ints() -> None:
    cap = RequestCapture(consumer="filesystem", hooks={"post_block": (0, 1)})
    assert cap.hooks == {"post_block": [0, 1]}


# -- GenerationRequest capture opt-in ----------------------------------------


def test_generation_request_capture_default_none() -> None:
    req = GenerationRequest(prompt="hi", max_tokens=8)
    assert req.capture is None


def test_generation_request_capture_roundtrip() -> None:
    cap = RequestCapture(consumer="filesystem", hooks={"post_block": [6]})
    req = GenerationRequest(prompt="hi", max_tokens=8, capture=cap)
    assert req.capture is cap
    assert req.capture.to_capture_field()["filesystem"]["hooks"] == {"post_block": [6]}


def test_generation_request_rejects_bad_capture_type() -> None:
    with pytest.raises(SteeringSpecError):
        GenerationRequest(prompt="hi", max_tokens=8, capture={"filesystem": {}})  # type: ignore[arg-type]
