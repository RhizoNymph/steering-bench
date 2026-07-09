"""capture-gated discovery + default capture methods raise EngineError.

Runs WITHOUT a GPU and WITHOUT any engine package installed by monkeypatching
the package-availability probe.
"""

from __future__ import annotations

import pytest

from steering_bench.engine import registry as reg
from steering_bench.engine.base import Capabilities, EngineError, SteeringEngine
from steering_bench.engine.capture import CaptureConsumerSpec
from steering_bench.engine.spec import GenerationRequest, GenerationResult


class _NoCaptureEngine(SteeringEngine):
    """Minimal engine without the capture capability."""

    name = "no_capture"
    capabilities = Capabilities(batching=True)

    def load(self, model_id: str, *, steering_config=None, **opts: object) -> None:
        pass

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [GenerationResult(output_tokens=1) for _ in requests]

    def memory_allocated_mb(self) -> float:
        return 0.0

    def teardown(self) -> None:
        pass


# -- capability-gated discovery ----------------------------------------------


def test_discover_capture_required_returns_vllm(monkeypatch, capsys) -> None:
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: True)
    classes = reg.discover(required=Capabilities(capture=True))
    names = {c.name for c in classes}
    assert "vllm" in names
    # None of the non-capture engines qualify.
    for name in ("transformerlens", "hf_baseline", "nnsight", "repeng", "pyvene"):
        assert name not in names
    out = capsys.readouterr().out
    assert "transformerlens: SKIPPED" in out


def test_discover_without_capture_includes_non_capture(monkeypatch) -> None:
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: True)
    names = {c.name for c in reg.discover()}
    assert {"vllm", "transformerlens", "hf_baseline"} <= names


def test_registry_capture_flags() -> None:
    by_name = {e.name: e for e in reg.ENGINE_REGISTRY}
    assert by_name["vllm"].capabilities.capture is True
    for name in ("transformerlens", "hf_baseline", "nnsight", "repeng", "pyvene"):
        assert by_name[name].capabilities.capture is False


# -- default capture methods raise EngineError -------------------------------


def test_configure_capture_default_raises() -> None:
    engine = _NoCaptureEngine()
    spec = CaptureConsumerSpec(name="logging")
    with pytest.raises(EngineError) as exc:
        engine.configure_capture([spec])
    assert "capture unsupported" in str(exc.value)
    assert "no_capture" in str(exc.value)


def test_capture_status_default_raises() -> None:
    engine = _NoCaptureEngine()
    with pytest.raises(EngineError):
        engine.capture_status()


def test_live_capture_consumers_default_raises() -> None:
    engine = _NoCaptureEngine()
    with pytest.raises(EngineError):
        engine.live_capture_consumers()
