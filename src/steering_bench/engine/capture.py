"""Capture-consumer declaration for the engine seam (CaptureEngine).

Capture lives entirely in the vLLM fork today. This module models the
consumer-spec configuration the public-API scripts pass to
``LLM(capture_consumers=[...])`` as a typed, immutable :class:`CaptureConsumerSpec`
so benchmarks stop hand-rolling config dicts and raw ``from vllm import`` calls.

The engine-level capture surface (declare consumers at load, introspect status)
is defined as default-raising methods on :class:`steering_bench.engine.base.SteeringEngine`
(mirroring ``register_module``): only engines advertising ``Capabilities.capture``
override them. The vLLM adapter is the sole implementer today.

Per-request capture opt-in (``SamplingParams(capture=...)`` shape) is modelled by
:class:`steering_bench.engine.spec.RequestCapture` on ``GenerationRequest`` so a
non-capturing request is unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Location = Literal["worker", "driver"]
Execution = Literal["sync", "async"]


class CaptureSpecError(ValueError):
    """Raised when a capture-consumer spec is structurally invalid."""


@dataclass(frozen=True)
class CaptureConsumerSpec:
    """One capture consumer declared at engine load.

    Models the config schema the scripts pass to ``LLM(capture_consumers=[...])``:
    an entry is either a config dict ``{"name", "params"[, "instance_name"]}``
    resolved by name from the ``vllm.capture_consumers`` entry points, or a
    pre-built driver-side consumer ``instance`` (e.g. ``RecordingDriverConsumer``)
    passed straight through.

    Fields:
      * ``name`` -- entry-point / identity name (always required, even for an
        instance, so results can label the arm).
      * ``params`` -- the consumer's construction params. By convention carries
        ``hooks: {hook: [layers]}``, ``positions``, and ``level`` for logging.
      * ``location`` -- ``"worker"`` (in the worker process) or ``"driver"``.
      * ``execution`` -- ``"sync"`` (per-step ``on_step``) or ``"async"``
        (finalize ``on_capture``). Metadata for the seam / result labelling; the
        fork infers the actual dispatch path from the consumer class.
      * ``instance_name`` -- disambiguates multiple consumers of the same
        ``name`` (the union-gather ``logging_3x`` arm).
      * ``instance`` -- an already-constructed driver-side consumer object,
        passed to ``LLM`` verbatim.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    location: Location = "worker"
    execution: Execution = "async"
    instance_name: str | None = None
    instance: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CaptureSpecError("CaptureConsumerSpec.name must be a non-empty str")
        if not isinstance(self.params, Mapping):
            raise CaptureSpecError(
                f"CaptureConsumerSpec.params must be a mapping, got {type(self.params)!r}"
            )
        if self.location not in ("worker", "driver"):
            raise CaptureSpecError(
                f"CaptureConsumerSpec.location must be 'worker'|'driver', got {self.location!r}"
            )
        if self.execution not in ("sync", "async"):
            raise CaptureSpecError(
                f"CaptureConsumerSpec.execution must be 'sync'|'async', got {self.execution!r}"
            )
        if self.instance_name is not None and (
            not isinstance(self.instance_name, str) or not self.instance_name.strip()
        ):
            raise CaptureSpecError(
                "CaptureConsumerSpec.instance_name must be a non-empty str or None"
            )

    def to_llm_config(self) -> Any:
        """The value to place in the ``LLM(capture_consumers=[...])`` list.

        A pre-built ``instance`` is passed through verbatim; otherwise a config
        dict ``{"name", "params"[, "instance_name"]}`` the fork resolves by name.
        """
        if self.instance is not None:
            return self.instance
        cfg: dict[str, Any] = {"name": self.name, "params": dict(self.params)}
        if self.instance_name is not None:
            cfg["instance_name"] = self.instance_name
        return cfg


def capture_consumers_arg(
    specs: list[CaptureConsumerSpec] | None,
) -> list[Any] | None:
    """Translate a spec list to the ``LLM(capture_consumers=...)`` argument.

    Returns ``None`` for an empty / unset list (the no-capture baseline), so the
    LLM constructor sees exactly what the raw scripts passed.
    """
    if not specs:
        return None
    return [spec.to_llm_config() for spec in specs]
