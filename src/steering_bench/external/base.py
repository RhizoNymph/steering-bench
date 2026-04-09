"""Base protocol for cross-library steering benchmarks.

All external library benchmarks implement SteeringBenchmark so the
runner script can treat them uniformly.
"""

from __future__ import annotations

import importlib.util
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SteeringBenchmark(Protocol):
    """Common interface for steering library benchmarks."""

    name: str

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        """Load model and configure steering.

        Args:
            model_id: HuggingFace model identifier.
            vector: Flat steering vector for a single layer.
            layer: Layer index to apply steering at.
            hook: Hook point name (e.g. "post_mlp").
        """
        ...

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        """Generate with steering applied, single request.

        Returns:
            Number of output tokens generated.
        """
        ...

    def generate_batch(
        self,
        prompts: list[str],
        vectors: list[list[float]],
        max_tokens: int,
    ) -> list[int]:
        """Generate with per-prompt steering, batch of N requests.

        Libraries that don't support batching should implement this
        as a sequential loop over generate_single().

        Returns:
            Number of output tokens per prompt.
        """
        ...

    def teardown(self) -> None:
        """Unload model and free GPU memory."""
        ...

    def memory_allocated_mb(self) -> float:
        """Current GPU memory usage in MB."""
        ...


def is_library_available(name: str) -> bool:
    """Check if a Python package is importable."""
    return importlib.util.find_spec(name) is not None


def gpu_memory_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def cleanup_gpu() -> None:
    """Force GPU memory cleanup between benchmarks."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
