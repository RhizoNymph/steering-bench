"""ServingEngine seam: the online/HTTP steering transport.

The synchronous ``SteeringEngine.generate(list) -> list`` ABC cannot express
online serving -- a long-lived API server driven by *streaming* completions
whose value is the per-token latency profile (TTFT / TPOT / ITL / E2EL), not a
single batch wall-time. This module models that transport as its own ABC with a
distinct lifecycle (``start_server`` / ``stop_server``) and an async streaming
driver.

Payload encoding is **owned by the adapter**, not the caller: the per-request
inline steering blob (base64 float32 packing) and the named-module register
payload are pure, unit-testable functions here (:func:`pack_steering_vectors`,
:func:`named_register_payload`, :func:`steering_extra_body`) so benchmark scripts
never hand-encode wire bytes. The vLLM implementation lives in
:mod:`steering_bench.engine.engines.vllm_serving`.

Nothing in this module imports ``openai`` / ``httpx`` / ``vllm`` at import scope;
the ABC and the encoders are pure. Concrete adapters import their HTTP stack
lazily inside method bodies.
"""

from __future__ import annotations

import abc
import base64
from dataclasses import dataclass, field
from typing import Any

from steering_bench.engine.base import Capabilities
from steering_bench.engine.spec import (
    GenerationRequest,
    NamedModuleRef,
    PhaseSteeringSpec,
    Steering,
    SteeringSpec,
)


class ServingError(RuntimeError):
    """Raised when a serving engine is misused or its server is unavailable."""


# -- per-request / metrics types ---------------------------------------------


@dataclass
class RequestResult:
    """Per-request measurement from one streaming completion.

    ``itl_ms`` holds inter-token latencies (gaps between successive decoded
    tokens). ``ttft_ms`` is the time to the first token, ``e2el_ms`` the
    end-to-end latency. ``error`` is set (and the timing fields left ``None``)
    when the request failed.
    """

    ttft_ms: float | None = None
    e2el_ms: float | None = None
    num_output_tokens: int = 0
    itl_ms: list[float] = field(default_factory=list)
    error: str | None = None


def compute_request_metrics(
    t0: float,
    token_times: list[float],
    end_time: float | None = None,
) -> RequestResult:
    """Compute a :class:`RequestResult` from raw token timestamps. Pure.

    ``t0`` is the request-start ``perf_counter`` value; ``token_times`` are the
    per-token arrival timestamps (same clock), in order. ``end_time`` is when the
    stream closed (defaults to the last token's timestamp).

    Formulas (identical to the legacy ``bench_serving.py`` driver):
      * ``ttft_ms``  = (first_token - t0) * 1000
      * ``itl_ms[i]``= (token[i] - token[i-1]) * 1000  for i >= 1
      * ``e2el_ms``  = (end_time - t0) * 1000
      * ``num_output_tokens`` = len(token_times)
    """
    result = RequestResult(num_output_tokens=len(token_times))
    if not token_times:
        result.e2el_ms = ((end_time if end_time is not None else t0) - t0) * 1000.0
        return result
    result.ttft_ms = (token_times[0] - t0) * 1000.0
    result.itl_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]
    last = end_time if end_time is not None else token_times[-1]
    result.e2el_ms = (last - t0) * 1000.0
    return result


def summarize_results(results: list[RequestResult]) -> dict[str, Any]:
    """Aggregate per-request results into a serving ``results`` block. Pure.

    TPOT is ``(e2el - ttft) / max(1, num_output_tokens - 1)`` per request. Each
    metric family is reported via :func:`steering_bench.timing.compute_stats`
    (minus the raw ``samples_ms``). Mirrors the legacy ``summarize`` exactly.
    """
    from steering_bench.timing import compute_stats

    ok = [r for r in results if r.error is None and r.ttft_ms is not None]
    errs = [r.error for r in results if r.error is not None]
    if not ok:
        return {"num_ok": 0, "num_err": len(errs), "errors": errs[:5]}

    ttft = [r.ttft_ms for r in ok]
    e2el = [r.e2el_ms for r in ok]
    all_itl = [v for r in ok for v in r.itl_ms]
    tpot = [
        (r.e2el_ms - r.ttft_ms) / max(1, r.num_output_tokens - 1)
        for r in ok
        if r.num_output_tokens > 1
    ]
    total_out = sum(r.num_output_tokens for r in ok)
    wall_s = max(e2el) / 1000.0 if e2el else 0.0

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {}
        d = compute_stats(xs).to_dict()
        return {k: v for k, v in d.items() if k != "samples_ms"}

    return {
        "num_ok": len(ok),
        "num_err": len(errs),
        "errors": errs[:5],
        "ttft_ms": _stats(ttft),
        "tpot_ms": _stats(tpot),
        "itl_ms": _stats(all_itl),
        "e2el_ms": _stats(e2el),
        "total_output_tokens": total_out,
        "offline_output_tps": total_out / wall_s if wall_s > 0 else 0.0,
    }


# -- adapter-owned payload encoding (pure, no vllm/openai/httpx) --------------


def pack_steering_vectors(spec: SteeringSpec) -> dict[str, dict[str, Any]]:
    """Pack a :class:`SteeringSpec` to the base64 float32 wire form. Pure.

    Produces ``{hook: {dtype, shape, layer_indices, data[, scales]}}`` where
    ``data`` is base64-encoded little-endian float32 bytes of the layer-stacked
    array (rows in ascending layer order). When the spec carries per-row
    ``scales``, each hook's slice (see :meth:`SteeringSpec.scales_for`) rides
    along as a ``scales`` list so the server exercises the per-row multiply path.

    This is the shape the HTTP ``steering_vectors`` field requires; the legacy
    list-of-floats JSON form is rejected server-side.
    """
    import numpy as np

    out: dict[str, dict[str, Any]] = {}
    for hook in spec.vectors:
        layer_indices = sorted(spec.vectors[hook])
        arr = np.stack(
            [np.asarray(spec.vectors[hook][i], dtype=np.float32) for i in layer_indices]
        )
        entry: dict[str, Any] = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "layer_indices": layer_indices,
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        scales = spec.scales_for(hook)
        if scales is not None:
            entry["scales"] = [float(x) for x in scales]
        out[hook] = entry
    return out


def steering_extra_body(steering: Steering) -> dict[str, Any] | None:
    """Map a request's steering field to the OpenAI ``extra_body`` dict. Pure.

    ``None`` -> no extra body (baseline / idle); a :class:`SteeringSpec` ->
    ``{"steering_vectors": <packed>}``; a :class:`NamedModuleRef` ->
    ``{"steering_name": name}`` (the server resolves the pre-registered module).
    """
    match steering:
        case None:
            return None
        case SteeringSpec():
            return {"steering_vectors": pack_steering_vectors(steering)}
        case NamedModuleRef():
            return {"steering_name": steering.name}
    raise TypeError(f"unsupported steering type: {type(steering)!r}")


def _spec_to_raw(spec: SteeringSpec) -> dict[str, dict[int, list[float]]]:
    return spec.to_vector_dict()


def named_register_payload(
    name: str,
    spec: SteeringSpec | PhaseSteeringSpec,
    *,
    prefill: SteeringSpec | None = None,
    decode: SteeringSpec | None = None,
) -> dict[str, Any]:
    """Build the ``POST /v1/steering/modules/register`` payload. Pure.

    Produces ``{name, vectors, prefill_vectors, decode_vectors}`` where
    ``vectors`` is the raw ``{hook: {layer: [floats]}}`` form (the register
    endpoint accepts raw vectors, unlike the per-request packed field).
    ``spec`` may be a :class:`SteeringSpec` (with ``prefill`` / ``decode`` passed
    as kwargs) or a :class:`PhaseSteeringSpec` bundling all three; the phase
    fields default to ``None``.
    """
    if isinstance(spec, PhaseSteeringSpec):
        base = spec.base
        prefill = prefill if prefill is not None else spec.prefill
        decode = decode if decode is not None else spec.decode
    else:
        base = spec
    return {
        "name": name,
        "vectors": _spec_to_raw(base),
        "prefill_vectors": _spec_to_raw(prefill) if prefill is not None else None,
        "decode_vectors": _spec_to_raw(decode) if decode is not None else None,
    }


# -- the ServingEngine ABC ----------------------------------------------------


@dataclass(frozen=True)
class ServingConfig:
    """Load-time configuration for an online serving server.

    ``enable_steering`` gates the ``--enable-steering`` flag; ``max_steering_configs``
    sizes the worker steering table; ``dev_mode`` / ``timing`` toggle the
    dev-only admin endpoints (named-module register + timing dump). ``enforce_eager``
    disables CUDA graph capture. Remaining server flags travel through
    ``extra_flags``.
    """

    enable_steering: bool = False
    max_steering_configs: int = 16
    dev_mode: bool = False
    timing: bool = False
    enforce_eager: bool = False
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    host: str = "127.0.0.1"
    port: int = 8765
    startup_timeout: float = 240.0
    extra_flags: tuple[str, ...] = ()


class ServingEngine(abc.ABC):
    """Abstract base for an online/HTTP steering serving adapter.

    Distinct from :class:`~steering_bench.engine.base.SteeringEngine`: the
    transport is a subprocess API server plus async *streaming* completions, and
    the measured quantity is the per-token latency profile. Adapters set ``name``
    / ``capabilities`` (with ``Capabilities.serving = True``) and implement the
    server lifecycle + async driver + admin endpoints below.
    """

    name: str = "unknown"
    capabilities: Capabilities = Capabilities(serving=True)

    @abc.abstractmethod
    def start_server(self, model_id: str, *, config: ServingConfig) -> None:
        """Launch the API server subprocess and block until it is healthy."""

    @abc.abstractmethod
    def stop_server(self) -> None:
        """Terminate the API server subprocess (graceful, then hard kill)."""

    @property
    @abc.abstractmethod
    def base_url(self) -> str:
        """OpenAI-compatible base URL of the running server (``.../v1``)."""

    @abc.abstractmethod
    async def run_request(self, request: GenerationRequest) -> RequestResult:
        """Drive one streaming completion, returning its measured result.

        The adapter owns steering encoding: it translates ``request.steering``
        via :func:`steering_extra_body` into the request's ``extra_body``.
        """

    async def run_workload(
        self, requests: list[GenerationRequest], concurrency: int
    ) -> list[RequestResult]:
        """Run ``requests`` with at most ``concurrency`` in flight, in order.

        Default implementation fans out :meth:`run_request` under a semaphore.
        """
        import asyncio

        sem = asyncio.Semaphore(concurrency)

        async def _guarded(req: GenerationRequest) -> RequestResult:
            async with sem:
                return await self.run_request(req)

        return await asyncio.gather(*[_guarded(r) for r in requests])

    async def warmup(
        self,
        requests: list[GenerationRequest],
        *,
        concurrency: int,
        drain_seconds: float = 0.5,
    ) -> None:
        """Fire a discarded warmup wave, then drain async background work.

        Primes Triton JIT / first-touch allocations, then (when
        ``drain_seconds > 0``) sleeps and fires a single 1-token soft-barrier
        request so fire-and-forget work queued during warmup lands before
        measurement. Mirrors the legacy warmup/drain barrier.
        """
        import asyncio

        if not requests:
            return
        await self.run_workload(requests, concurrency)
        if drain_seconds > 0:
            await asyncio.sleep(drain_seconds)
            barrier = GenerationRequest(prompt=requests[0].prompt, max_tokens=1)
            await self.run_workload([barrier], 1)

    @abc.abstractmethod
    async def register_named_module(
        self,
        name: str,
        spec: SteeringSpec | PhaseSteeringSpec,
        *,
        prefill: SteeringSpec | None = None,
        decode: SteeringSpec | None = None,
    ) -> None:
        """Register a named steering module via the dev-mode admin endpoint."""

    @abc.abstractmethod
    async def dump_and_reset_timings(self, mode: str, *, quiet: bool = False) -> None:
        """Dump + reset the server's per-worker steering-timing accumulators.

        No-op unless the server was started with ``timing`` and ``dev_mode``.
        ``quiet`` resets silently (used between warmup and measurement).
        """

    # -- identity ------------------------------------------------------------

    def version(self) -> str:
        return "unknown"

    def commit(self) -> str | None:
        return None

    def identity(self) -> dict[str, str | None]:
        return {"name": self.name, "version": self.version(), "commit": self.commit()}


# -- serving-engine registry (few adapters; vLLM only today) ------------------


@dataclass(frozen=True)
class ServingEngineEntry:
    """A registered serving adapter, described without importing it."""

    name: str
    required_package: str | None
    module_path: str
    class_name: str


SERVING_ENGINE_REGISTRY: list[ServingEngineEntry] = [
    ServingEngineEntry(
        name="vllm",
        required_package="vllm",
        module_path="steering_bench.engine.engines.vllm_serving",
        class_name="VllmServingEngine",
    ),
]


def get_serving_engine(name: str) -> type[ServingEngine]:
    """Import and return the serving adapter class named ``name``."""
    import importlib

    for entry in SERVING_ENGINE_REGISTRY:
        if entry.name == name:
            module = importlib.import_module(entry.module_path)
            return getattr(module, entry.class_name)
    known = ", ".join(e.name for e in SERVING_ENGINE_REGISTRY)
    raise ServingError(f"unknown serving engine {name!r}; known: {known}")
