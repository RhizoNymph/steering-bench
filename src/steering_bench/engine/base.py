"""Engine abstraction: capabilities + the ``SteeringEngine`` ABC.

An engine is one adapter class. It advertises a fixed ``Capabilities`` set so
the registry can filter engines to those able to serve a given benchmark, and
implements a tiny lifecycle: ``load`` -> ``generate`` -> ``teardown`` plus
memory and identity introspection.

GPU helpers are reused from the existing ``external`` package rather than
duplicated, so behavior stays consistent with the legacy adapters.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields

from steering_bench.engine.spec import GenerationRequest, GenerationResult
from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class EngineError(RuntimeError):
    """Raised when an engine is asked to do something it cannot support."""


@dataclass(frozen=True)
class Capabilities:
    """What an engine can do, as a set of independent boolean axes.

    Used both to describe an engine and to express a requirement.  A required
    capability of ``False`` means "don't care".
    """

    batching: bool = False
    named_modules: bool = False
    multi_layer: bool = False
    multi_hook: bool = False
    capture: bool = False

    def satisfies(self, required: Capabilities) -> bool:
        """True if ``self`` provides every capability ``required`` demands.

        A ``required`` field of ``False`` imposes no constraint.
        """
        return all(
            getattr(self, f.name) or not getattr(required, f.name)
            for f in fields(required)
        )


class SteeringEngine(abc.ABC):
    """Abstract base for a steering-capable inference engine adapter.

    Subclasses set the ``name`` and ``capabilities`` class attributes and
    implement the abstract lifecycle methods.  ``version``/``commit`` default to
    an unknown identity and are overridden by concrete adapters.
    """

    name: str = "unknown"
    capabilities: Capabilities = Capabilities()

    @abc.abstractmethod
    def load(self, model_id: str, **opts: object) -> None:
        """Load ``model_id`` and configure steering. Engine-specific ``opts``."""

    @abc.abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Run a batch of requests, returning one result per request in order."""

    @abc.abstractmethod
    def memory_allocated_mb(self) -> float:
        """Current GPU memory usage in MB."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Unload the model and free GPU memory."""

    # -- identity (overridden by adapters) -----------------------------------

    def version(self) -> str:
        """Version string of the underlying engine."""
        return "unknown"

    def commit(self) -> str | None:
        """Commit hash of the underlying engine, if available."""
        return None

    # -- shared GPU helpers (reused from external.base) ----------------------

    @staticmethod
    def _gpu_memory_mb() -> float:
        return gpu_memory_mb()

    @staticmethod
    def _cleanup_gpu() -> None:
        cleanup_gpu()
