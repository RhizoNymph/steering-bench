"""repeng steering engine adapter.

Ports ``external/repeng_bench.py``: repeng wraps the model in a ``ControlModel``
over a band of layers around the target and applies a ``ControlVector`` via
``set_control`` -- forward-pass vector addition, architecturally closest to
vLLM's approach.  repeng has no continuous batching, so a batch is generated
sequentially.

**Load-time-steering caching.**  ``set_control`` re-parameterizes the model, so
the control object is built lazily on first ``generate`` and cached keyed on the
request's ``SteeringSpec`` (via :func:`spec_cache_key`).  It is rebuilt only when
a request's spec differs from the cached one.  A shared-vector workload therefore
pays the build once, while a changing-vector workload legitimately incurs -- and
measures -- the re-parameterization cost on every change.

``repeng``, ``transformers`` and ``torch`` are imported lazily inside methods so
this module is importable without them.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import (
    Capabilities,
    EngineError,
    SteeringConfig,
    SteeringEngine,
)
from steering_bench.engine.spec import (
    GenerationRequest,
    GenerationResult,
    NamedModuleRef,
    SteeringSpec,
)

# Half-width of the layer band controlled around the target layer (matches the
# legacy adapter's ``range(layer-2, layer+3)``).
_BAND_BELOW = 2
_BAND_ABOVE = 3

# A hashable cache key derived from a SteeringSpec's content.
SpecKey = tuple[tuple[str, tuple[tuple[int, tuple[float, ...]], ...]], ...]


def spec_cache_key(spec: SteeringSpec) -> SpecKey:
    """A hashable key capturing a spec's full content.

    Two structurally-identical specs produce equal keys (cache hit); any change
    to a hook, layer, or vector value produces a different key (rebuild).  Pure:
    no heavy-lib import.
    """
    return tuple(
        (hook, tuple(sorted((layer, vec) for layer, vec in layers.items())))
        for hook, layers in spec.vectors.items()
    )


def control_layers_for(layer: int, num_layers: int) -> list[int]:
    """The band of layers ``ControlModel`` should control around ``layer``."""
    return list(range(max(0, layer - _BAND_BELOW), min(num_layers, layer + _BAND_ABOVE)))


def resolve_single(spec: SteeringSpec) -> tuple[str, int, tuple[float, ...]]:
    """Validate that ``spec`` is single-hook/single-layer and unpack it.

    Pure helper (no torch); raises ``EngineError`` on unsupported specs.
    """
    if spec.is_multi_hook():
        raise EngineError(f"repeng supports a single hook, got {spec.hooks()}")
    hook = spec.hooks()[0]
    layers = spec.layers(hook)
    if len(layers) != 1:
        raise EngineError(
            f"repeng supports a single control layer, got {layers} for hook {hook!r}"
        )
    layer = layers[0]
    return hook, layer, spec.vectors[hook][layer]


class RepengSteeringEngine(SteeringEngine):
    """Adapter over ``repeng.ControlModel`` + ``set_control``."""

    name = "repeng"
    capabilities = Capabilities(
        batching=False,  # sequential over the batch
        named_modules=False,
        multi_layer=True,  # controls a band of layers around the target
        multi_hook=False,
        capture=False,
    )

    def __init__(self) -> None:
        self._base_model: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None  # the active ControlModel
        self._cached_key: SpecKey | None = None
        self._num_layers: int = 0

    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        # No fork-specific steering knobs; steering_config is ignored (no-op).
        del steering_config
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
        self._num_layers = self._base_model.config.num_hidden_layers

    def _ensure_control(self, spec: SteeringSpec) -> None:
        """Build/refresh the ControlModel for ``spec``, caching on its content."""
        import torch
        from repeng import ControlModel, ControlVector

        key = spec_cache_key(spec)
        if key == self._cached_key and self._model is not None:
            return  # cache hit: shared vector already parameterized.

        _hook, layer, vec = resolve_single(spec)
        if self._model is not None:
            self._model.reset()
        control_layers = control_layers_for(layer, self._num_layers)
        self._model = ControlModel(self._base_model, control_layers)
        control_vector = ControlVector(
            model_type=self._base_model.config.model_type,
            directions={layer: torch.tensor(list(vec), dtype=torch.float16)},
        )
        self._model.set_control(control_vector, coeff=1.0)
        self._cached_key = key

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        import torch

        if self._base_model is None or self._tokenizer is None:
            raise RuntimeError("RepengSteeringEngine.generate called before load()")

        results: list[GenerationResult] = []
        for req in requests:
            match req.steering:
                case SteeringSpec() as spec:
                    self._ensure_control(spec)
                case NamedModuleRef():
                    raise EngineError("repeng does not support named steering modules")
                case None:
                    raise EngineError("repeng requires an inline steering spec")
                case other:  # pragma: no cover - guarded by types
                    raise EngineError(f"unsupported steering: {type(other)!r}")

            inputs = self._tokenizer(req.prompt, return_tensors="pt").to(
                self._model.model.device
            )
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            results.append(GenerationResult(output_tokens=outputs.shape[1] - input_len))

        return results

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        if self._model is not None:
            self._model.reset()
        self._model = None
        self._base_model = None
        self._tokenizer = None
        self._cached_key = None
        self._cleanup_gpu()

    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("repeng")
        except PackageNotFoundError:
            return "unknown"
