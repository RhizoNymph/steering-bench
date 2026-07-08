"""Typed domain model for the engine seam.

The core idea is to make invalid steering interventions unrepresentable.
Engines consume a small, canonical vocabulary:

- ``SteeringSpec``  -- an inline intervention: a nested mapping of
  ``hook_point -> layer_idx -> vector``.  This mirrors the format emitted
  by :func:`steering_bench.vectors.random_steering_vectors` and consumed by
  vLLM's ``SamplingParams.steering_vectors``, but is validated and immutable.
- ``NamedModuleRef`` -- a reference to a pre-registered steering module by name.
- ``GenerationRequest`` -- a prompt + token budget + optional steering.
- ``GenerationResult`` -- what an engine reports back for one request.

All validation happens in ``__post_init__`` and raises ``SteeringSpecError``
(a ``ValueError`` subclass) with an actionable message.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class SteeringSpecError(ValueError):
    """Raised when a steering spec / request is structurally invalid."""


# Canonical inline intervention: hook_point -> layer_idx -> vector.
VectorMap = dict[str, dict[int, tuple[float, ...]]]


@dataclass(frozen=True)
class SteeringSpec:
    """An inline steering intervention.

    ``vectors`` maps ``hook_point -> layer_idx -> vector``.  Inner vectors are
    normalized to tuples of floats on construction, so a spec is effectively
    immutable and cheap to compare.

    Invariants (enforced in ``__post_init__``):
      * at least one hook, and every hook has at least one layer,
      * every vector for a given hook has the same, non-zero length.
    """

    vectors: VectorMap

    def __post_init__(self) -> None:
        raw = self.vectors
        if not isinstance(raw, Mapping) or not raw:
            raise SteeringSpecError(
                "SteeringSpec.vectors must be a non-empty mapping of "
                "hook -> {layer: vector}"
            )

        normalized: VectorMap = {}
        for hook, layers in raw.items():
            if not isinstance(hook, str) or not hook:
                raise SteeringSpecError(f"hook point must be a non-empty str, got {hook!r}")
            if not isinstance(layers, Mapping) or not layers:
                raise SteeringSpecError(f"hook {hook!r} has no layers")

            norm_layers: dict[int, tuple[float, ...]] = {}
            dims: set[int] = set()
            for layer_idx, vec in layers.items():
                if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
                    raise SteeringSpecError(
                        f"hook {hook!r} layer {layer_idx!r}: vector must be a sequence"
                    )
                tup = tuple(float(x) for x in vec)
                dims.add(len(tup))
                norm_layers[int(layer_idx)] = tup

            if len(dims) != 1:
                raise SteeringSpecError(
                    f"hook {hook!r} has ragged vector lengths: {sorted(dims)}"
                )
            if next(iter(dims)) == 0:
                raise SteeringSpecError(f"hook {hook!r} has zero-length vectors")

            normalized[hook] = norm_layers

        object.__setattr__(self, "vectors", normalized)

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_vector_dict(
        cls, d: Mapping[str, Mapping[int, Sequence[float]]]
    ) -> SteeringSpec:
        """Build from the ``{hook: {layer: [floats]}}`` dict that
        :func:`steering_bench.vectors.random_steering_vectors` emits."""
        return cls(vectors={hook: dict(layers) for hook, layers in d.items()})  # type: ignore[arg-type]

    @classmethod
    def single(cls, hook: str, layer: int, vector: Sequence[float]) -> SteeringSpec:
        """Build a single-hook, single-layer spec."""
        return cls(vectors={hook: {layer: tuple(float(x) for x in vector)}})

    # -- introspection -------------------------------------------------------

    def hooks(self) -> tuple[str, ...]:
        """Hook points this spec covers, in insertion order."""
        return tuple(self.vectors)

    def layers(self, hook: str) -> tuple[int, ...]:
        """Sorted layer indices covered for ``hook``."""
        if hook not in self.vectors:
            raise SteeringSpecError(f"unknown hook {hook!r}; have {self.hooks()}")
        return tuple(sorted(self.vectors[hook]))

    def dim(self, hook: str) -> int:
        """Vector length for ``hook`` (uniform across its layers)."""
        if hook not in self.vectors:
            raise SteeringSpecError(f"unknown hook {hook!r}; have {self.hooks()}")
        any_layer = next(iter(self.vectors[hook].values()))
        return len(any_layer)

    def is_multi_hook(self) -> bool:
        return len(self.vectors) > 1

    def is_multi_layer(self) -> bool:
        return any(len(layers) > 1 for layers in self.vectors.values())

    def is_single_hook_single_layer(self) -> bool:
        return not self.is_multi_hook() and not self.is_multi_layer()

    def to_vector_dict(self) -> dict[str, dict[int, list[float]]]:
        """Read back as a plain ``{hook: {layer: [floats]}}`` dict."""
        return {
            hook: {layer: list(vec) for layer, vec in layers.items()}
            for hook, layers in self.vectors.items()
        }


@dataclass(frozen=True)
class NamedModuleRef:
    """Reference to a pre-registered named steering module.

    ``scale`` is the per-reference multiplier applied to the registered
    module's vectors at request time.  It defaults to ``1.0`` (apply the
    module as registered).  Adapters that support named modules encode the
    reference as a ``(name, scale)`` pair (the vLLM fork's
    ``SamplingParams.steering_module_ref`` format).
    """

    name: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SteeringSpecError("NamedModuleRef.name must be a non-empty str")
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise SteeringSpecError(
                f"NamedModuleRef.scale must be finite, got {self.scale!r}"
            )
        object.__setattr__(self, "scale", scale)


# What an engine is asked to apply for a request: an inline spec, a named
# module reference, or nothing (plain generation baseline).
Steering = SteeringSpec | NamedModuleRef | None


@dataclass(frozen=True)
class GenerationRequest:
    """A single generation request with optional steering."""

    prompt: str
    max_tokens: int
    steering: Steering = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise SteeringSpecError(
                f"GenerationRequest.max_tokens must be > 0, got {self.max_tokens}"
            )


@dataclass(frozen=True)
class GenerationResult:
    """Result of one generation request.

    Minimal by design; ``text`` is optional so latency/throughput benchmarks
    need not materialize decoded strings.

    ``output_tokens_exact`` records whether ``output_tokens`` is an exact,
    per-request count.  It defaults to ``True`` (engines that count honestly).
    An engine whose batch path cannot recover per-prompt output lengths -- e.g.
    nnsight's pseudo-batch, which reports the ``max_tokens`` placeholder -- sets
    it ``False`` so downstream consumers do not treat the figure as precise.
    """

    output_tokens: int
    text: str | None = None
    output_tokens_exact: bool = True

    def __post_init__(self) -> None:
        if self.output_tokens < 0:
            raise SteeringSpecError(
                f"GenerationResult.output_tokens must be >= 0, got {self.output_tokens}"
            )
