"""Capture-consumer + dynamic-steering benchmark consumers.

One consumer per non-baseline arm of ``scripts/bench_dynamic_steering.py``.
Each does a fixed, minimal amount of policy work so the cross-arm comparison
isolates the cost of the capture/steering *transport* (the tier), not of any
real policy. The steering arms keep steering active for the whole batch from
the first decode step on, so every decode token is steered every step
(steady-state overhead) while the per-step decision cost stays negligible.

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
                  GLOBAL monitor (probe + tier) ONCE; the monitor op then gates
                  the tier per token *inside the forward* (same-token). A
                  saturated negative threshold holds the gate ~1 so steering
                  stays active every step — this measures the in-graph monitor
                  overhead with steering live, not a disengaged probe.
  steer_override  ``enable_steering``; sync consumer routes every live decode
                  request to its OWN dynamic-pool override row (one
                  ``RequestSteeringOverride`` per request, deduped, pruned to
                  the live set). Measures the per-request override pool
                  (dynamic rows) at steady state.
  steer_rowmon    ``enable_steering`` + ``enable_row_monitor``; same as
                  ``steer_override`` PLUS a PER-ROW ``SteeringMonitorUpdate``
                  keyed by ``req_id`` for each request (emitted in the same
                  action list AFTER that request's override so the runner can
                  resolve ``req_id -> dyn row`` from the override applied
                  earlier in the same in-order pass). A saturated negative
                  threshold holds each row's gate ~1 so steering stays live —
                  this measures the per-row monitor cost, not disengagement.

Multi-site steering: ``_Base`` accepts ``layers`` (list[int]) and ``hooks``
(list[str]); the steering arms build one vector per ``(hook, layer)`` site
(distinct seeded vectors, equal norm). The single-site ``layer``/``hook``
params still work (back-compat). The monitor (global or per-row) always sits at
ONE configured ``monitor_layer``/``monitor_hook`` site (default: the first
steer site).

Registered as ``vllm.capture_consumers`` entry points in ``pyproject.toml`` so
the worker-side (sync/async) consumers can be constructed in each worker
process by name.

Activation introspection: every consumer registers itself in the process-global
``_LIVE_CONSUMERS`` list on construction. With
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the worker runs in the benchmark cell's own
process, so the cell can read each consumer's live counters
(``iter_live_consumers``) to assert the arm is genuinely active after warmup —
the capture arms have no worker-side status RPC, so this is how their step
counters are checked. The steering arms are additionally verified via the
``get_dynamic_steering_status`` collective RPC (tier/monitor active,
applied/rejected counters).
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

# The block-output hook. Renamed from the old ``post_mlp`` (a stale name is no
# longer a member of ``SteeringHookPoint`` on the current branch: a capture spec
# or steering update carrying it is rejected, silently disabling the arm — which
# is exactly what the per-cell activation assertion exists to catch).
_HOOK = "post_block"


# Process-global registry of every live bench consumer instance, so the
# in-process benchmark cell can introspect worker-side activation counters.
_LIVE_CONSUMERS: list["_Base"] = []


def iter_live_consumers() -> list["_Base"]:
    """Live bench consumer instances in this process (multiprocessing=0)."""
    return list(_LIVE_CONSUMERS)


def reset_live_consumers() -> None:
    """Clear the registry (each fresh cell process starts empty anyway)."""
    _LIVE_CONSUMERS.clear()


def _unit(hidden: int, seed: int) -> np.ndarray:
    """Deterministic (seeded) unit vector — identical across ranks/runs."""
    v = np.random.default_rng(seed).standard_normal(hidden).astype(np.float32)
    return np.ascontiguousarray(v / float(np.linalg.norm(v)), dtype=np.float32)


def _parse_int_list(value: Any) -> list[int]:
    """Accept an int, a list of ints, or a comma string of ints."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split(",") if str(x).strip() != ""]


def _parse_str_list(value: Any, default: str) -> list[str]:
    """Accept a str, a list of str, or a comma string of hook names."""
    if value is None:
        return [default]
    if isinstance(value, (list, tuple)):
        out = [str(x) for x in value]
        return out or [default]
    parts = [x.strip() for x in str(value).split(",") if x.strip() != ""]
    return parts or [default]


class _Base:
    """Shared config loading + global capture spec for the bench consumers.

    Steer sites are the Cartesian product of ``hooks`` x ``layers`` (each a
    distinct seeded vector of equal norm). The monitor site is a single
    ``(monitor_hook, monitor_layer)`` — defaulting to the first steer site.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        self._hidden = model_config.get_hidden_size() if model_config else None

        # Multi-site: ``layers``/``hooks`` (lists) generalize ``layer``/``hook``.
        layers = _parse_int_list(params.get("layers"))
        if not layers:
            layers = [int(params["layer"])]
        hooks = _parse_str_list(params.get("hooks"), str(params.get("hook", _HOOK)))
        self._layers = layers
        self._hooks = hooks
        self._norm = float(params.get("norm", 8.0))

        # Monitor site — one (hook, layer); defaults to the first steer site.
        self._mon_hook = str(params.get("monitor_hook", hooks[0]))
        self._mon_layer = int(params.get("monitor_layer", layers[0]))

        # One distinct seeded vector per (hook, layer) site, equal norm.
        self._sites: list[tuple[str, int]] = [
            (h, layer) for h in hooks for layer in layers
        ]
        self._vectors: dict[str, dict[int, np.ndarray]] = {}
        if self._hidden:
            for idx, (h, layer) in enumerate(self._sites):
                # seed >= 1 so the vectors never collide with the probe (seed 0).
                vec = _unit(self._hidden, idx + 1) * self._norm
                self._vectors.setdefault(h, {})[layer] = vec
        self._probe = _unit(self._hidden, 0) if self._hidden else None

        self._steps = 0
        _LIVE_CONSUMERS.append(self)

    # ---- capture spec: cover every steer site plus the monitor site --------
    def global_capture_spec(self) -> CaptureSpec:
        hooks: dict[str, list[int]] = {}
        for h, layer in self._sites:
            hooks.setdefault(h, [])
            if layer not in hooks[h]:
                hooks[h].append(layer)
        hooks.setdefault(self._mon_hook, [])
        if self._mon_layer not in hooks[self._mon_hook]:
            hooks[self._mon_hook].append(self._mon_layer)
        return CaptureSpec(hooks=hooks, positions="all_generated")

    def status(self) -> dict[str, Any]:
        return {"steps": self._steps, "sites": len(self._sites)}

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def _tier_update(self, source: str) -> SteeringVectorUpdate:
        return SteeringVectorUpdate(
            vectors=self._vectors, phase="decode", source=source
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
        if queue is None or self._submitted or not self._vectors:
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
        if self._emitted or not self._vectors:
            return None
        self._emitted = True
        return [self._tier_update("bench_steer_sync")]


class BenchSteerDynamic(_Base):
    """In-graph GLOBAL monitor: install a probe + tier once; the monitor op
    gates the tier per token inside the forward (same-token). Saturated
    negative threshold => gate ~1 => steering stays active (measures monitor
    overhead)."""

    execution: ClassVar[Literal["sync"]] = "sync"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", -1.0e9))
        self._sharpness = float(params.get("sharpness", 8.0))
        self._installed = False

    def on_step(self, view: Any) -> list[Any] | None:
        self._steps += 1
        if self._installed or not self._vectors:
            return None
        self._installed = True
        return [
            self._tier_update("bench_steer_dynamic"),
            SteeringMonitorUpdate(
                hook=self._mon_hook,
                layer=self._mon_layer,
                probe=self._probe,
                threshold=self._threshold,
                sharpness=self._sharpness,
                source="bench_steer_dynamic",
            ),
        ]


class _PerRequestBase(_Base):
    """Sync consumer: route every live decode request to its own dynamic-pool
    override row (one ``RequestSteeringOverride`` per request), deduped and
    pruned to the live set. Needs ``max_dynamic_steering_configs >=
    batch_size``. Subclasses add a per-request companion action (e.g. the
    per-row monitor) via :meth:`_companion`."""

    execution: ClassVar[Literal["sync"]] = "sync"
    _SOURCE: ClassVar[str] = "bench_steer_override"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._emitted: set[str] = set()

    def _companion(self, req_id: str) -> list[Any]:
        """Per-request action(s) emitted AFTER the override in the same list."""
        return []

    def on_step(self, view: Any) -> list[Any] | None:
        self._steps += 1
        if not self._vectors:
            return None
        # Prune to the requests actually present this step so the emitted set
        # does not grow without bound (req_ids are never reused, so a still
        # co-scheduled batch is never dropped mid-run).
        present = {req.req_id for req in view.requests}
        self._emitted &= present

        actions: list[Any] = []
        for req in view.requests:
            if req.phase != "decode" or req.req_id in self._emitted:
                continue
            self._emitted.add(req.req_id)
            # Override FIRST so a per-request companion (row monitor / scale)
            # can resolve the freshly registered dyn row in the same in-order
            # apply pass.
            actions.append(
                RequestSteeringOverride(
                    req_id=req.req_id,
                    vectors=self._vectors,
                    source=self._SOURCE,
                )
            )
            actions.extend(self._companion(req.req_id))
        return actions or None


class BenchSteerOverride(_PerRequestBase):
    """Per-request override pool, no monitor (the row-monitor control arm)."""

    _SOURCE = "bench_steer_override"


class BenchSteerRowmon(_PerRequestBase):
    """Per-request override PLUS a per-row in-graph monitor keyed by
    ``req_id`` (requires ``enable_row_monitor``). Saturated negative threshold
    => each row's gate ~1 => steering stays live (measures the per-row monitor
    cost, not disengagement)."""

    _SOURCE = "bench_steer_rowmon"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", -1.0e9))
        self._sharpness = float(params.get("sharpness", 8.0))

    def _companion(self, req_id: str) -> list[Any]:
        return [
            SteeringMonitorUpdate(
                hook=self._mon_hook,
                layer=self._mon_layer,
                probe=self._probe,
                threshold=self._threshold,
                sharpness=self._sharpness,
                req_id=req_id,
                source=self._SOURCE,
            )
        ]
