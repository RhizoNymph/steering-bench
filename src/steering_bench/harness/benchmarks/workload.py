"""Shared workload helpers for harness benchmarks."""

from __future__ import annotations

from steering_bench.engine.spec import GenerationRequest, SteeringSpec
from steering_bench.harness.models import get_model_config
from steering_bench.vectors import random_steering_vectors


def steering_spec_for(model: str, layer: int, hook: str, seed: int = 42) -> SteeringSpec:
    """Build a single-hook/single-layer ``SteeringSpec`` for ``model``.

    Resolves the model's dimensions via :func:`get_model_config`, generates a
    reproducible random vector at ``layer`` for ``hook``, and wraps it in a spec.
    """
    cfg = get_model_config(model)
    vectors = random_steering_vectors(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        hook_points=[hook],
        scale=0.1,
        seed=seed,
    )
    return SteeringSpec.single(hook, layer, vectors[hook][layer])


def make_prompt(prompt_len: int = 64) -> str:
    """A dummy prompt of roughly ``prompt_len`` tokens (~1.3 tokens/word)."""
    words_needed = max(1, int(prompt_len / 1.3))
    return " ".join(["hello"] * words_needed)


def steered_requests(
    model: str,
    layer: int,
    hook: str,
    max_tokens: int,
    batch_size: int,
    prompt_len: int = 64,
) -> list[GenerationRequest]:
    """A batch of ``batch_size`` identically-steered generation requests."""
    spec = steering_spec_for(model, layer, hook)
    prompt = make_prompt(prompt_len)
    return [
        GenerationRequest(prompt=prompt, max_tokens=max_tokens, steering=spec)
        for _ in range(batch_size)
    ]
