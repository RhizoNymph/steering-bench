"""Engine abstraction seam for the steering benchmark harness.

Public vocabulary:
  * ``SteeringSpec`` / ``NamedModuleRef`` / ``GenerationRequest`` /
    ``GenerationResult`` -- the typed domain model (see ``spec``).
  * ``SteeringEngine`` / ``Capabilities`` / ``EngineError`` -- the engine ABC
    and its capability descriptor (see ``base``).
  * ``ENGINE_REGISTRY`` / ``EngineEntry`` / ``discover`` -- capability-aware
    engine discovery (see ``registry``).

Concrete adapters live in ``steering_bench.engine.engines`` and are imported
lazily by ``discover`` so this package is importable with no backends present.
"""

from __future__ import annotations

from steering_bench.engine.base import (
    Capabilities,
    EngineError,
    SteeringConfig,
    SteeringEngine,
)
from steering_bench.engine.registry import (
    ENGINE_REGISTRY,
    EngineEntry,
    discover,
    is_package_available,
)
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    Steering,
    SteeringSpec,
    SteeringSpecError,
)

__all__ = [
    "ENGINE_REGISTRY",
    "Capabilities",
    "EngineEntry",
    "EngineError",
    "GenerationRequest",
    "GenerationResult",
    "NamedModuleRef",
    "Steering",
    "SteeringConfig",
    "SteeringEngine",
    "SteeringSpec",
    "SteeringSpecError",
    "discover",
    "is_package_available",
]
