"""vLLM steering engine adapter.

The SteeringSpec -> vLLM-native translation lives in pure module-level
functions (``spec_to_native``, ``named_ref_to_kwargs``, ``steering_kwargs``)
that do NOT import vllm, so they are unit-testable without vllm installed.
The engine methods import vllm lazily inside their bodies, mirroring the legacy
``external/vllm_single.py`` / ``vllm_batched.py`` adapters.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities, SteeringEngine
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    Steering,
    SteeringSpec,
)

# vLLM load defaults, matching the legacy external adapters.
_DEFAULT_LOAD_OPTS: dict[str, Any] = {
    "enable_steering": True,
    "max_steering_configs": 4,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048,
}


def spec_to_native(spec: SteeringSpec) -> dict[str, dict[int, list[float]]]:
    """Translate a ``SteeringSpec`` to vLLM's native inline vector format.

    Produces ``{hook: {layer: [floats]}}`` suitable for
    ``SamplingParams(steering_vectors=...)``. Pure: no vllm import.
    """
    return spec.to_vector_dict()


def named_ref_to_kwargs(ref: NamedModuleRef) -> dict[str, Any]:
    """Translate a ``NamedModuleRef`` to ``SamplingParams`` kwargs."""
    return {"steering_module_ref": ref.name}


def steering_kwargs(steering: Steering) -> dict[str, Any]:
    """Map a request's steering field to ``SamplingParams`` kwargs. Pure."""
    match steering:
        case None:
            return {}
        case SteeringSpec():
            return {"steering_vectors": spec_to_native(steering)}
        case NamedModuleRef():
            return named_ref_to_kwargs(steering)
    raise TypeError(f"unsupported steering type: {type(steering)!r}")


class VllmSteeringEngine(SteeringEngine):
    """Adapter over the vLLM steering fork's ``LLM`` batch API."""

    name = "vllm"
    capabilities = Capabilities(
        batching=True,
        named_modules=True,
        multi_layer=True,
        multi_hook=True,
        capture=True,
    )

    def __init__(self) -> None:
        self._llm: Any | None = None

    def load(self, model_id: str, **opts: object) -> None:
        from vllm import LLM

        load_opts = {**_DEFAULT_LOAD_OPTS, **opts}
        self._llm = LLM(model=model_id, **load_opts)

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
