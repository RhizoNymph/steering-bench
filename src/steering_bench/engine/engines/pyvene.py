"""pyvene steering engine adapter.

Ports ``external/pyvene_bench.py``: pyvene drives a config-declared intervention
graph.  An ``IntervenableModel`` built from an ``IntervenableConfig`` +
``AdditionIntervention`` at the target layer/component adds the steering vector
during generation.  pyvene does not optimize batch dispatch, so a batch is
generated sequentially.

**Load-time-steering caching.**  Building the ``IntervenableModel`` (and the
intervention source tensor) re-parameterizes the model, so it is built lazily on
first ``generate`` and cached keyed on the request's ``SteeringSpec`` (via
:func:`spec_cache_key`).  It is rebuilt only when a request's spec differs from
the cached one: a shared-vector workload pays the build once, a changing-vector
workload legitimately measures the re-parameterization cost.

``pyvene``, ``transformers`` and ``torch`` are imported lazily inside methods so
this module is importable without them.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities, EngineError, SteeringEngine
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    SteeringSpec,
)

# hook_point name -> pyvene component the intervention attaches to.
_COMPONENT_MAP: dict[str, str] = {
    "pre_attn": "block_input",
    "post_attn": "block_output",
    "post_mlp": "block_output",
}
_DEFAULT_COMPONENT = "block_output"

# A hashable cache key derived from a SteeringSpec's content.
SpecKey = tuple[tuple[str, tuple[tuple[int, tuple[float, ...]], ...]], ...]


def spec_cache_key(spec: SteeringSpec) -> SpecKey:
    """A hashable key capturing a spec's full content.

    Equal for structurally-identical specs (cache hit); different when any hook,
    layer, or vector value changes (rebuild).  Pure: no heavy-lib import.
    """
    return tuple(
        (hook, tuple(sorted((layer, vec) for layer, vec in layers.items())))
        for hook, layers in spec.vectors.items()
    )


def component_for(hook: str) -> str:
    """Map a hook-point name to the pyvene component to intervene on."""
    return _COMPONENT_MAP.get(hook, _DEFAULT_COMPONENT)


def resolve_single(spec: SteeringSpec) -> tuple[str, int, tuple[float, ...]]:
    """Validate that ``spec`` is single-hook/single-layer and unpack it.

    Pure helper (no torch); raises ``EngineError`` on unsupported specs.
    """
    if spec.is_multi_hook():
        raise EngineError(f"pyvene supports a single hook, got {spec.hooks()}")
    hook = spec.hooks()[0]
    layers = spec.layers(hook)
    if len(layers) != 1:
        raise EngineError(
            f"pyvene supports a single layer, got {layers} for hook {hook!r}"
        )
    layer = layers[0]
    return hook, layer, spec.vectors[hook][layer]


class PyveneSteeringEngine(SteeringEngine):
    """Adapter over ``pyvene.IntervenableModel`` + ``AdditionIntervention``."""

    name = "pyvene"
    capabilities = Capabilities(
        batching=False,  # sequential over the batch
        named_modules=False,
        multi_layer=False,
        multi_hook=False,
        capture=False,
    )

    def __init__(self) -> None:
        self._base_model: Any | None = None
        self._tokenizer: Any | None = None
        self._intervenable: Any | None = None
        self._intervention_vec: Any | None = None
        self._cached_key: SpecKey | None = None

    def load(self, model_id: str, **opts: object) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = opts.pop("torch_dtype", torch.float16)
        device_map = opts.pop("device_map", "auto")
        self._base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map, **opts
        )

    def _ensure_intervenable(self, spec: SteeringSpec) -> None:
        """Build/refresh the IntervenableModel for ``spec``, caching on content."""
        import pyvene
        import torch

        key = spec_cache_key(spec)
        if key == self._cached_key and self._intervenable is not None:
            return  # cache hit: shared vector already parameterized.

        hook, layer, vec = resolve_single(spec)
        config = pyvene.IntervenableConfig(
            representations=[
                pyvene.RepresentationConfig(
                    layer=layer,
                    component=component_for(hook),
                    intervention_type=pyvene.AdditionIntervention,
                )
            ]
        )
        self._intervenable = pyvene.IntervenableModel(config, self._base_model)
        tensor = torch.tensor(list(vec), dtype=torch.float16, device="cuda")
        # Shape for intervention: (batch, seq, hidden).
        self._intervention_vec = tensor.unsqueeze(0).unsqueeze(0)
        self._cached_key = key

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        import torch

        if self._base_model is None or self._tokenizer is None:
            raise RuntimeError("PyveneSteeringEngine.generate called before load()")

        results: list[GenerationResult] = []
        for req in requests:
            match req.steering:
                case SteeringSpec() as spec:
                    self._ensure_intervenable(spec)
                case NamedModuleRef():
                    raise EngineError("pyvene does not support named steering modules")
                case None:
                    raise EngineError("pyvene requires an inline steering spec")
                case other:  # pragma: no cover - guarded by types
                    raise EngineError(f"unsupported steering: {type(other)!r}")

            inputs = self._tokenizer(req.prompt, return_tensors="pt").to(
                self._base_model.device
            )
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self._intervenable.generate(
                    {"input_ids": inputs["input_ids"]},
                    [self._intervention_vec],
                    max_new_tokens=req.max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
            results.append(
                GenerationResult(output_tokens=output_ids.shape[1] - input_len)
            )

        return results

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        self._intervenable = None
        self._intervention_vec = None
        self._base_model = None
        self._tokenizer = None
        self._cached_key = None
        self._cleanup_gpu()

    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pyvene")
        except PackageNotFoundError:
            return "unknown"
