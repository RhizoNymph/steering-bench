"""vLLM steering engine adapter.

The SteeringSpec -> vLLM-native translation lives in pure module-level
functions (``spec_to_native``, ``named_ref_to_kwargs``, ``steering_kwargs``)
that do NOT import vllm, so they are unit-testable without vllm installed.
The engine methods import vllm lazily inside their bodies, mirroring the legacy
``external/vllm_single.py`` / ``vllm_batched.py`` adapters.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities, SteeringConfig, SteeringEngine
from steering_bench.engine.capture import CaptureConsumerSpec, capture_consumers_arg
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    RequestCapture,
    Steering,
    SteeringSpec,
)

# vLLM load defaults for knobs the typed SteeringConfig does not cover.
_DEFAULT_LOAD_OPTS: dict[str, Any] = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048,
}
# Default load-time steering configuration when a caller passes none.
_DEFAULT_STEERING_CONFIG = SteeringConfig(
    enable_steering=True, max_steering_configs=4, enable_prefix_caching=True
)


def spec_to_native(spec: SteeringSpec) -> dict[str, dict[int, list[float]]]:
    """Translate a ``SteeringSpec`` to vLLM's native inline vector format.

    Produces ``{hook: {layer: [floats]}}`` suitable for
    ``SamplingParams(steering_vectors=...)``. Pure: no vllm import.
    """
    return spec.to_vector_dict()


def named_ref_to_kwargs(ref: NamedModuleRef) -> dict[str, Any]:
    """Translate a ``NamedModuleRef`` to ``SamplingParams`` kwargs.

    The vLLM steering fork expects ``steering_module_ref`` as a
    ``(name, scale)`` tuple, not a bare name string.
    """
    return {"steering_module_ref": (ref.name, ref.scale)}


def _coerce_vector(vec: Any) -> list[float]:
    """Coerce a single vector (list / tuple / numpy array) to a list of floats."""
    tolist = getattr(vec, "tolist", None)
    if callable(tolist):  # numpy array / torch tensor
        vec = tolist()
    return [float(x) for x in vec]


def named_payload_from_spec(
    spec: SteeringSpec,
    *,
    prefill: SteeringSpec | None = None,
    decode: SteeringSpec | None = None,
) -> dict[str, Any]:
    """Build the ``register_steering_modules`` payload for one module.

    Produces ``{"vectors": {hook: {layer: [floats]}}}`` from a ``SteeringSpec``,
    coercing numpy arrays / tuples to plain lists.  Pure: no vllm import.  When
    ``prefill`` / ``decode`` specs are given, their vectors are attached under
    ``"prefill_vectors"`` / ``"decode_vectors"`` for phase-split serving.
    """
    payload: dict[str, Any] = {"vectors": _spec_vectors(spec)}
    if prefill is not None:
        payload["prefill_vectors"] = _spec_vectors(prefill)
    if decode is not None:
        payload["decode_vectors"] = _spec_vectors(decode)
    return payload


def _spec_vectors(spec: SteeringSpec) -> dict[str, dict[int, list[float]]]:
    return {
        hook: {int(layer): _coerce_vector(vec) for layer, vec in layers.items()}
        for hook, layers in spec.vectors.items()
    }


def steering_kwargs(steering: Steering) -> dict[str, Any]:
    """Map a request's steering field to ``SamplingParams`` kwargs. Pure."""
    match steering:
        case None:
            return {}
        case SteeringSpec():
            # The offline inline format (``steering_vectors``) has no per-row
            # scale field — only the serving packer honors ``SteeringSpec.scales``.
            # Refuse rather than silently drop them; bake scales into the vectors
            # or use a named module / the serving path instead.
            if steering.scales is not None:
                raise ValueError(
                    "offline vLLM inline steering cannot express per-row scales "
                    "(SteeringSpec.scales); bake the scales into the vectors, or "
                    "use a named module / the serving path."
                )
            return {"steering_vectors": spec_to_native(steering)}
        case NamedModuleRef():
            return named_ref_to_kwargs(steering)
    raise TypeError(f"unsupported steering type: {type(steering)!r}")


def capture_kwargs(capture: RequestCapture | None) -> dict[str, Any]:
    """Map a request's per-request capture opt-in to ``SamplingParams`` kwargs.

    ``None`` -> no ``capture`` key (unaffected request); otherwise the nested
    ``{consumer: {...}}`` dict the fork's ``SamplingParams.capture`` expects. Pure.
    """
    if capture is None:
        return {}
    return {"capture": capture.to_capture_field()}


class VllmSteeringEngine(SteeringEngine):
    """Adapter over the vLLM steering fork's ``LLM`` batch API."""

    name = "vllm"
    capabilities = Capabilities(
        batching=True,
        named_modules=True,
        multi_layer=True,
        multi_hook=True,
        capture=True,
        prefix_cache=True,
        config_capacity=True,
    )

    def __init__(self) -> None:
        self._llm: Any | None = None
        self._capture_specs: list[CaptureConsumerSpec] = []

    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        from vllm import LLM

        cfg = steering_config or _DEFAULT_STEERING_CONFIG
        load_opts: dict[str, Any] = {
            **_DEFAULT_LOAD_OPTS,
            "enable_steering": cfg.enable_steering,
            "max_steering_configs": cfg.max_steering_configs,
            "enable_prefix_caching": cfg.enable_prefix_caching,
            **opts,
        }
        consumers = capture_consumers_arg(self._capture_specs)
        if consumers is not None:
            load_opts["capture_consumers"] = consumers
        self._llm = LLM(model=model_id, **load_opts)

    def _collective_rpc(self, method: str) -> Any:
        """Call a worker collective RPC, preferring the ``LLM`` passthrough."""
        if self._llm is None:
            raise RuntimeError("VllmSteeringEngine collective_rpc before load()")
        rpc = getattr(self._llm, "collective_rpc", None)
        if rpc is None:
            rpc = self._llm.llm_engine.collective_rpc
        return rpc(method)

    # -- capture (CaptureEngine surface) -------------------------------------

    def configure_capture(self, specs: list[CaptureConsumerSpec]) -> None:
        """Store capture-consumer specs to install at the next ``load``."""
        self._capture_specs = list(specs)

    def capture_status(self) -> list[dict[str, Any]]:
        """Per-worker ``get_dynamic_steering_status`` payloads (one per worker)."""
        res = self._collective_rpc("get_dynamic_steering_status")
        return list(res) if isinstance(res, (list, tuple)) else [res]

    def live_capture_consumers(self) -> list[Any]:
        """Live bench capture-consumer instances in this process (mp=0)."""
        from steering_bench.capture_consumers.bench_consumers import (
            iter_live_consumers,
        )

        return iter_live_consumers()

    def register_module(
        self,
        name: str,
        spec: SteeringSpec,
        *,
        replace: bool = True,
        prefill: SteeringSpec | None = None,
        decode: SteeringSpec | None = None,
    ) -> None:
        if self._llm is None:
            raise RuntimeError(
                "VllmSteeringEngine.register_module called before load()"
            )
        payload = named_payload_from_spec(spec, prefill=prefill, decode=decode)
        self._llm.llm_engine.collective_rpc(
            "register_steering_modules",
            kwargs={"modules": {name: payload}, "replace": replace},
        )

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("VllmSteeringEngine.generate called before load()")

        prompts: list[str] = []
        sampling_params: list[Any] = []
        for req in requests:
            prompts.append(req.prompt)
            sampling_params.append(
                SamplingParams(
                    max_tokens=req.max_tokens,
                    temperature=0.0,
                    **steering_kwargs(req.steering),
                    **capture_kwargs(req.capture),
                )
            )

        outputs = self._llm.generate(prompts, sampling_params)
        return [
            GenerationResult(output_tokens=len(out.outputs[0].token_ids))
            for out in outputs
        ]

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        self._llm = None
        self._cleanup_gpu()

    def version(self) -> str:
        import vllm

        return vllm.__version__

    def commit(self) -> str | None:
        import vllm

        return getattr(vllm, "__commit__", None)
