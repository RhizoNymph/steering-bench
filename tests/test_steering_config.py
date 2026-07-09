"""Tests for the typed load-time SteeringConfig surface (Phase 3)."""

from __future__ import annotations

import pytest

from steering_bench.engine.base import Capabilities, SteeringConfig, SteeringEngine
from steering_bench.engine.registry import ENGINE_REGISTRY
from steering_bench.engine.spec import GenerationRequest, GenerationResult


def test_defaults() -> None:
    cfg = SteeringConfig()
    assert cfg.enable_steering is True
    assert cfg.max_steering_configs == 4
    assert cfg.enable_prefix_caching is True


def test_frozen_and_validated() -> None:
    with pytest.raises(ValueError):
        SteeringConfig(max_steering_configs=-1)


class _FakeNonVllmEngine(SteeringEngine):
    """A non-vLLM engine: no prefix cache, no config capacity."""

    name = "fake_nonvllm"
    capabilities = Capabilities(batching=True)

    def __init__(self) -> None:
        self.received_config: SteeringConfig | None = None
        self.backend_opts: dict[str, object] = {}

    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        # Translate what we support (nothing), no-op the vLLM-only knobs.
        self.received_config = steering_config
        self.backend_opts = dict(opts)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [GenerationResult(output_tokens=1) for _ in requests]

    def memory_allocated_mb(self) -> float:
        return 0.0

    def teardown(self) -> None:
        pass


def test_non_vllm_engine_no_ops_vllm_only_knobs() -> None:
    engine = _FakeNonVllmEngine()
    cfg = SteeringConfig(
        enable_steering=True, max_steering_configs=64, enable_prefix_caching=False
    )
    engine.load("fake/model", steering_config=cfg)
    # The engine received the config but does NOT surface max_steering_configs /
    # prefix caching to its backend (it advertises neither capability).
    assert engine.received_config is cfg
    assert "max_steering_configs" not in engine.backend_opts
    assert "enable_prefix_caching" not in engine.backend_opts
    assert engine.capabilities.prefix_cache is False
    assert engine.capabilities.config_capacity is False


def test_capability_flags_additive_default_false() -> None:
    caps = Capabilities()
    assert caps.prefix_cache is False
    assert caps.config_capacity is False


def test_vllm_advertises_config_capabilities() -> None:
    vllm_entry = next(e for e in ENGINE_REGISTRY if e.name == "vllm")
    assert vllm_entry.capabilities.prefix_cache is True
    assert vllm_entry.capabilities.config_capacity is True


def test_backward_compatible_load_without_steering_config() -> None:
    engine = _FakeNonVllmEngine()
    engine.load("fake/model")
    assert engine.received_config is None
