"""Steering vector generation utilities.

Generates reproducible random steering vectors for benchmarking.
These are NOT semantically meaningful -- they're for measuring
performance overhead, not steering quality.
"""

from __future__ import annotations

import torch


def random_steering_vectors(
    hidden_size: int,
    num_layers: int,
    hook_points: list[str] | None = None,
    scale: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, dict[int, list[float]]]:
    """Generate random steering vectors in the SamplingParams format.

    Args:
        hidden_size: Model hidden dimension.
        num_layers: Number of decoder layers.
        hook_points: Which hook points to generate for.
            Defaults to ["post_mlp"].
        scale: Scale factor for random values (keep small to avoid
            destabilizing generation).
        seed: Random seed for reproducibility.
        device: Device for generation (vectors are returned as lists).

    Returns:
        Dict in SamplingParams.steering_vectors format:
        ``{hook_point: {layer_idx: [floats]}}``
    """
    if hook_points is None:
        hook_points = ["post_mlp"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    result: dict[str, dict[int, list[float]]] = {}
    for hp in hook_points:
        layers: dict[int, list[float]] = {}
        for layer_idx in range(num_layers):
            vec = torch.randn(hidden_size, generator=gen, device=device) * scale
            layers[layer_idx] = vec.tolist()
        result[hp] = layers
    return result


def random_steering_vectors_diverse(
    hidden_size: int,
    num_layers: int,
    num_configs: int,
    hook_points: list[str] | None = None,
    scale: float = 0.1,
    base_seed: int = 42,
    device: str = "cpu",
) -> list[dict[str, dict[int, list[float]]]]:
    """Generate multiple distinct steering vector configs.

    Each config uses a different seed (base_seed + i) so they're
    all unique but reproducible.

    Returns:
        List of num_configs steering vector dicts.
    """
    return [
        random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=hook_points,
            scale=scale,
            seed=base_seed + i,
            device=device,
        )
        for i in range(num_configs)
    ]
