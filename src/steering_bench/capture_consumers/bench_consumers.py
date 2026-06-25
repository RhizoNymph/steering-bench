"""Capture-consumer + dynamic-steering benchmark consumers.

One consumer per non-baseline arm of ``scripts/bench_dynamic_steering.py``.
Each does a fixed, minimal amount of policy work so the cross-arm comparison
isolates the cost of the capture/steering *transport* (the tier), not of any
real policy. The steering arms keep a GLOBAL decode tier active for the whole
batch from the first decode step on, so every decode token is steered every
step (steady-state overhead) while the per-step decision cost stays negligible.

Arms / tiers (all worker-side, global capture spec, no client spec):

  capture_async   async consumer; global capture spec at ``(hook, layer)``;
                  ``on_capture`` (fires at request finalize) only counts. No
                  steering. The preexisting async capture path.
  capture_sync    sync consumer; ``on_step`` runs every step and only counts,
                  emitting nothing. No steering. Isolates the per-step sync
                  ``on_step`` machinery against async ``on_capture``.
  steer_async     ``enable_steering``; async consumer submits ONE global
                  decode-tier ``SteeringVectorUpdate`` through the action queue
                  (drained on a later step). Steady-state: tier active.
  steer_sync      ``enable_steering``; sync consumer emits ONE global
                  decode-tier update from ``on_step`` (exactly-one-step
                  latency), then only counts each step.
  steer_dynamic   ``enable_steering``; sync consumer installs an in-graph
                  monitor (probe + tier) ONCE; the monitor op then gates the
                  tier per token *inside the forward* (same-token). A saturated
                  negative threshold holds the gate ~1 so steering stays active
                  every step — this measures the in-graph monitor overhead with
                  steering live, not a disengaged probe.

Registered as ``vllm.capture_consumers`` entry points in ``pyproject.toml`` so
the worker-side (sync/async) consumers can be constructed in each worker
process by name.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureSpec
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringVectorUpdate,
    get_steering_action_queue,
)

_HOOK = "post_block"


def _unit(hidden: int, seed: int) -> np.ndarray:
    """Deterministic (seeded) unit vector — identical across ranks/runs."""
    v = np.random.default_rng(seed).standard_normal(hidden).astype(np.float32)
    return np.ascontiguousarray(v / float(np.linalg.norm(v)), dtype=np.float32)


class _Base:
    """Shared config loading + global capture spec for the bench consumers."""

    location: ClassVar[Literal["worker"]] = "worker"

    @classmethod
    def declared_graphsafe_keys(cls, params: dict) -> list:
        # Sync consumers read the StepCaptureView directly; no per-request
        # graph-safe pre-buffering needed. Matches CaptureConsumer default.
        return []
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        self._hidden = model_config.get_hidden_size() if model_config else None
        self._layer = int(params["layer"])
        self._hook = str(params.get("hook", _HOOK))
        self._norm = float(params.get("norm", 8.0))
        self._probe = _unit(self._hidden, 0) if self._hidden else None
        self._vec = (
            _unit(self._hidden, 1) * self._norm if self._hidden else None
        )
        self._steps = 0

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={self._hook: [self._layer]}, positions="all_generated"
        )

    def status(self) -> dict[str, Any]:
        return {"steps": self._steps}

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def _tier_update(self, source: str) -> SteeringVectorUpdate:
        return SteeringVectorUpdate(
            vectors={self._hook: {self._layer: self._vec}},
            phase="decode",
            source=source,
        )


# --------------------------------------------------------------------------
# Capture-only arms (no steering)
# --------------------------------------------------------------------------


class BenchCaptureAsync(_Base, CaptureConsumer):
    """Async capture-only: ``on_capture`` fires at request finalize."""

    def on_capture(self, key: Any, tensor: Any, sidecar: dict[str, Any]) -> None:
        self._steps += 1


class BenchCaptureSync(_Base):
    """Sync capture-only: ``on_step`` runs every step, emits nothing."""

    execution: ClassVar[Literal["sync"]] = "sync"

    def on_step(self, view: Any) -> None:
        self._steps += 1
        return None


# --------------------------------------------------------------------------
# Steering arms (enable_steering; global tier kept active)
# --------------------------------------------------------------------------


class BenchSteerAsync(_Base, CaptureConsumer):
    """Async steering: submit one global decode-tier update via the queue."""

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._submitted = False

    def on_capture(self, key: Any, tensor: Any, sidecar: dict[str, Any]) -> None:
        self._steps += 1
        queue = get_steering_action_queue()
        if queue is None or self._submitted or self._vec is None:
            return
        if queue.submit(self._tier_update("bench_steer_async")):
            self._submitted = True


class BenchSteerSync(_Base):
    """Sync steering: emit one global decode-tier update from ``on_step``."""

    execution: ClassVar[Literal["sync"]] = "sync"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._emitted = False

    def on_step(self, view: Any) -> list[Any] | None:
        self._steps += 1
        if self._emitted or self._vec is None:
            return None
        self._emitted = True
        return [self._tier_update("bench_steer_sync")]


class BenchSteerDynamic(_Base):
    """In-graph steering: install a probe + tier once; the monitor op gates the
    tier per token inside the forward (same-token). Saturated negative
    threshold ⇒ gate ~1 ⇒ steering stays active (measures monitor overhead)."""

    execution: ClassVar[Literal["sync"]] = "sync"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", -1.0e9))
        self._sharpness = float(params.get("sharpness", 8.0))
        self._installed = False

    def on_step(self, view: Any) -> list[Any] | None:
        self._steps += 1
        if self._installed or self._vec is None:
            return None
        self._installed = True
        return [
            self._tier_update("bench_steer_dynamic"),
            SteeringMonitorUpdate(
                hook=self._hook,
                layer=self._layer,
                probe=self._probe,
                threshold=self._threshold,
                sharpness=self._sharpness,
                source="bench_steer_dynamic",
            ),
        ]



class BenchSteerPerRequest(_Base):
    """Per-request dynamic steering via the override pool. Instead of one
    global tier, install a DISTINCT ``RequestSteeringOverride`` for each
    request the first decode step it appears (distinct dyn_id row + distinct
    vector). Measures the per-request routing overhead (distinct rows,
    ``steering_index`` rebuild, ``req_id -> dyn_id`` resolution) under the
    override pool, vs the global-tier ``steer_dynamic`` arm."""

    execution: ClassVar[Literal["sync"]] = "sync"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._seen: set[str] = set()
        self._next_seed = 1000

    def on_step(self, view: Any) -> list[Any] | None:
        self._steps += 1
        if self._hidden is None:
            return None
        actions: list[Any] = []
        for req in getattr(view, "requests", []):
            rid = getattr(req, "req_id", None)
            if rid is None or getattr(req, "phase", None) != "decode":
                continue
            if rid in self._seen:
                continue
            self._seen.add(rid)
            vec = _unit(self._hidden, self._next_seed) * self._norm
            self._next_seed += 1
            actions.append(
                RequestSteeringOverride(
                    req_id=rid,
                    vectors={self._hook: {self._layer: vec}},
                    source="bench_steer_per_request",
                )
            )
        return actions or None
