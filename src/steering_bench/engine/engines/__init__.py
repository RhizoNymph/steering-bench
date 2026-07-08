"""Concrete steering engine adapters.

Both adapters import their heavy backend lazily inside method bodies, so
importing this package is cheap and dependency-free.
"""

from __future__ import annotations

from steering_bench.engine.engines.transformerlens import (
    TransformerLensSteeringEngine,
)
from steering_bench.engine.engines.vllm import (
    VllmSteeringEngine,
    named_ref_to_kwargs,
    spec_to_native,
    steering_kwargs,
)

__all__ = [
    "TransformerLensSteeringEngine",
    "VllmSteeringEngine",
    "named_ref_to_kwargs",
    "spec_to_native",
    "steering_kwargs",
]
