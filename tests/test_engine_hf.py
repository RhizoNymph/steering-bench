"""Pure-logic tests for the HF baseline engine adapter (no transformers)."""

from __future__ import annotations

import sys

from steering_bench.engine.engines.hf import HFSteeringEngine


def test_import_does_not_pull_transformers() -> None:
    # The adapter module must be importable without transformers/torch.
    assert "steering_bench.engine.engines.hf" in sys.modules


def test_capabilities_declaration() -> None:
    caps = HFSteeringEngine.capabilities
    assert caps.batching is True
    assert caps.named_modules is False
    assert caps.multi_layer is False
    assert caps.multi_hook is False
    assert caps.capture is False


def test_name_is_registry_consistent() -> None:
    assert HFSteeringEngine.name == "hf_baseline"


def test_generate_before_load_raises() -> None:
    engine = HFSteeringEngine()
    try:
        engine.generate([])
    except RuntimeError as exc:
        assert "before load" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")
