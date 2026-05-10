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
    layer_subset: list[int] | None = None,
) -> dict[str, dict[int, list[float]]]:
    """Generate random steering vectors in the SamplingParams format.

    Args:
        hidden_size: Model hidden dimension.
        num_layers: Number of decoder layers (used as upper bound when
            ``layer_subset`` is None).
        hook_points: Which hook points to generate for.
            Defaults to ["post_mlp"].
        scale: Scale factor for random values (keep small to avoid
            destabilizing generation).
        seed: Random seed for reproducibility.
        device: Device for generation (vectors are returned as lists).
        layer_subset: Explicit list of layer indices to populate.  When
            None (default), every layer in ``range(num_layers)`` is
            populated (the legacy "all layers" behavior).  Use this to
            sweep how many layers a request steers.

    Returns:
        Dict in SamplingParams.steering_vectors format:
        ``{hook_point: {layer_idx: [floats]}}``
    """
    if hook_points is None:
        hook_points = ["post_mlp"]

    if layer_subset is None:
        layer_indices = list(range(num_layers))
    else:
        bad = [i for i in layer_subset if i < 0 or i >= num_layers]
        if bad:
            raise ValueError(
                f"layer_subset indices out of range [0, {num_layers}): {bad}"
            )
        layer_indices = list(layer_subset)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    result: dict[str, dict[int, list[float]]] = {}
    for hp in hook_points:
        layers: dict[int, list[float]] = {}
        for layer_idx in layer_indices:
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
    layer_subset: list[int] | None = None,
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
            layer_subset=layer_subset,
        )
        for i in range(num_configs)
    ]


def even_layer_subset(num_layers: int, count: int) -> list[int]:
    """Return ``count`` evenly-spaced layer indices in ``[0, num_layers)``.

    Used by the modes-matrix bench to sweep "fraction of layers steered"
    in a deterministic way regardless of model depth.  Picks indices via
    ``round(i * (num_layers - 1) / (count - 1))`` and dedupes (so e.g.
    ``count > num_layers`` collapses to all layers).
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")
    if count >= num_layers:
        return list(range(num_layers))
    if count == 1:
        return [num_layers // 2]
    out: list[int] = []
    for i in range(count):
        idx = round(i * (num_layers - 1) / (count - 1))
        if idx not in out:
            out.append(idx)
    return out
