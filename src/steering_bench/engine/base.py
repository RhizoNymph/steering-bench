"""Engine abstraction: capabilities + the ``SteeringEngine`` ABC.

An engine is one adapter class. It advertises a fixed ``Capabilities`` set so
the registry can filter engines to those able to serve a given benchmark, and
implements a tiny lifecycle: ``load`` -> ``generate`` -> ``teardown`` plus
memory and identity introspection.

GPU / package helpers used by every adapter live here as the single source of
truth (they were previously in the retired ``external.base`` module).
"""

from __future__ import annotations

import abc
import importlib.util
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    SteeringSpec,
)

if TYPE_CHECKING:
    from steering_bench.engine.capture import CaptureConsumerSpec


class EngineError(RuntimeError):
    """Raised when an engine is asked to do something it cannot support."""


@dataclass(frozen=True)
class SteeringConfig:
    """Typed load-time steering configuration.

    A small, engine-agnostic surface so callers stop passing opaque ``**opts``
    for load-time steering knobs.  Engines translate the fields they support and
    **no-op the rest**: ``enable_steering`` is the only universally meaningful
    axis; ``max_steering_configs`` (worker steering-table capacity) and
    ``enable_prefix_caching`` are vLLM-fork concepts that non-vLLM engines
    ignore.  Advertise support via ``Capabilities.config_capacity`` /
    ``Capabilities.prefix_cache``.
    """

    enable_steering: bool = True
    max_steering_configs: int = 4
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.max_steering_configs < 0:
            raise ValueError(
                "SteeringConfig.max_steering_configs must be >= 0, got "
                f"{self.max_steering_configs}"
            )


# -- shared GPU / package helpers (single definition, formerly external.base) --


def gpu_memory_mb() -> float:
    """Current GPU memory allocated in MB (0.0 on CPU-only hosts)."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def cleanup_gpu() -> None:
    """Force GPU memory cleanup between benchmarks."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_library_available(name: str) -> bool:
    """Whether a Python package is importable."""
    return importlib.util.find_spec(name) is not None


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
    #: Honors ``SteeringConfig.enable_prefix_caching`` at load time.
    prefix_cache: bool = False
    #: Honors ``SteeringConfig.max_steering_configs`` (a bounded worker-side
    #: steering-config table) at load time.
    config_capacity: bool = False
    #: Exposes an online/HTTP serving transport (the ``ServingEngine`` seam:
    #: subprocess API server + async streaming completions). Additive; the vLLM
    #: serving adapter is the sole implementer today.
    serving: bool = False
    #: Exposes an activation-patching / causal-tracing sweep (the
    #: ``PatchSweepEngine`` seam: patch the clean run's residual into the corrupt
    #: run at every ``(layer, position)`` cell and grade the answer token). A
    #: distinct axis from generate / capture / serving; the TransformerLens and
    #: vLLM patch-sweep adapters implement it.
    patch_sweep: bool = False

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
    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        """Load ``model_id`` and configure steering.

        ``steering_config`` carries the typed load-time steering knobs
        (``enable_steering`` / ``max_steering_configs`` /
        ``enable_prefix_caching``).  An engine translates the fields it supports
        and no-ops the rest; ``None`` means "use engine defaults".  Remaining
        engine-specific settings still travel through ``opts``.
        """

    def register_module(
        self,
        name: str,
        spec: SteeringSpec,
        *,
        replace: bool = True,
        prefill: SteeringSpec | None = None,
        decode: SteeringSpec | None = None,
    ) -> None:
        """Register a named steering module so requests can reference it by name.

        Default implementation raises ``EngineError``: only engines advertising
        ``Capabilities.named_modules`` override this.  ``prefill`` / ``decode``
        allow distinct prefill/decode-phase vectors for serving; both default to
        ``None`` (use ``spec`` for every phase).
        """
        raise EngineError(f"named modules unsupported by {self.name}")

    # -- capture (CaptureEngine surface) -------------------------------------
    #
    # Capture is modelled as default-raising methods here (mirroring
    # ``register_module``) rather than a separate ABC, so that ANY engine
    # subclass answers ``EngineError`` -- not ``AttributeError`` -- when asked to
    # capture without the capability. Only engines advertising
    # ``Capabilities.capture`` override these; the vLLM adapter is the sole
    # implementer today. Selection is gated via
    # ``discover(required=Capabilities(capture=True))``.

    def configure_capture(self, specs: list[CaptureConsumerSpec]) -> None:
        """Declare capture consumers to install at ``load``.

        Call before ``load``: the engine stores the specs and constructs the
        capture manager (``LLM(capture_consumers=...)``) during ``load``. Default
        raises ``EngineError``; only capture-capable engines override.
        """
        raise EngineError(f"capture unsupported by {self.name}")

    def capture_status(self) -> list[dict[str, Any]]:
        """Per-worker capture / dynamic-steering status for activation asserts.

        Wraps the fork's ``collective_rpc("get_dynamic_steering_status")`` and
        returns one dict per worker. Default raises ``EngineError``.
        """
        raise EngineError(f"capture unsupported by {self.name}")

    def live_capture_consumers(self) -> list[Any]:
        """In-process live capture-consumer instances (multiprocessing=0).

        The capture arms have no worker status RPC; with the worker in-process
        the benchmark reads each consumer's step counters here. Default raises
        ``EngineError``.
        """
        raise EngineError(f"capture unsupported by {self.name}")

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

    def identity(self) -> dict[str, str | None]:
        """First-class engine identity block for the result schema.

        Canonical cross-engine identifier written to a result's top-level
        ``engine`` key via :func:`steering_bench.output.write_result`.
        """
        return {"name": self.name, "version": self.version(), "commit": self.commit()}

    # -- shared GPU helpers (module-level functions above) -------------------

    @staticmethod
    def _gpu_memory_mb() -> float:
        return gpu_memory_mb()

    @staticmethod
    def _cleanup_gpu() -> None:
        cleanup_gpu()
