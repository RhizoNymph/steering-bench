"""The default register_module raises EngineError (Phase 3)."""

from __future__ import annotations

import pytest

from steering_bench.engine.base import Capabilities, EngineError, SteeringEngine
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    SteeringSpec,
)


class _NoNamedEngine(SteeringEngine):
    """Minimal engine without the named_modules capability."""

    name = "no_named"
    capabilities = Capabilities(batching=True)

    def load(self, model_id: str, *, steering_config=None, **opts: object) -> None:
        pass

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [GenerationResult(output_tokens=1) for _ in requests]

    def memory_allocated_mb(self) -> float:
        return 0.0

    def teardown(self) -> None:
        pass


def test_default_register_module_raises_engine_error() -> None:
    engine = _NoNamedEngine()
    spec = SteeringSpec.single("post_block", 0, [1.0, 2.0])
    with pytest.raises(EngineError) as exc:
        engine.register_module("m", spec)
    assert "named modules unsupported" in str(exc.value)
    assert "no_named" in str(exc.value)


def test_default_register_module_error_mentions_engine_name() -> None:
    engine = _NoNamedEngine()
    spec = SteeringSpec.single("post_block", 0, [1.0])
    with pytest.raises(EngineError):
        engine.register_module("m", spec, replace=False)
