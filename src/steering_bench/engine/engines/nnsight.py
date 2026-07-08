"""nnsight steering engine adapter.

Ports ``external/nnsight_bench.py``: nnsight uses deferred execution via trace
contexts, adding the steering vector to a decoder layer's residual output
(``model.model.layers[layer].output[0]``).  It supports a *pseudo*-batch -- one
``generate`` trace with a per-prompt ``invoke`` -- but that path cannot recover
per-prompt output-token counts, so those results are marked
``output_tokens_exact=False`` (see ``GenerationResult``).

Only a single-hook/single-layer inline ``SteeringSpec`` is supported (residual
add at the layer output); ``nnsight`` and ``torch`` are imported lazily.
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

# Every supported hook point maps to the same nnsight target: an additive
# intervention on the decoder layer's residual output.  nnsight has no distinct
# attn-out / resid-pre split at this level, so the hook name only selects the
# (single) layer to steer.
_SUPPORTED_HOOKS: frozenset[str] = frozenset(
    {"pre_attn", "post_attn", "post_mlp", "post_block"}
)


def resolve_single(spec: SteeringSpec) -> tuple[str, int, tuple[float, ...]]:
    """Validate that ``spec`` is single-hook/single-layer and unpack it.

    Pure helper (no nnsight/torch); raises ``EngineError`` on unsupported specs.
    """
    if spec.is_multi_hook():
        raise EngineError(f"nnsight supports a single hook, got {spec.hooks()}")
    hook = spec.hooks()[0]
    layers = spec.layers(hook)
    if len(layers) != 1:
        raise EngineError(
            f"nnsight supports a single layer, got {layers} for hook {hook!r}"
        )
    layer = layers[0]
    return hook, layer, spec.vectors[hook][layer]


def layer_path(layer: int) -> str:
    """The nnsight residual-output access path for ``layer``."""
    return f"model.layers.{layer}.output[0]"


def batch_placeholder_results(
    requests: list[GenerationRequest],
) -> list[GenerationResult]:
    """Per-prompt results for the pseudo-batch path.

    The batched trace cannot recover per-prompt output lengths, so each result
    carries the ``max_tokens`` placeholder and is flagged inexact.  Pure: no
    nnsight/torch import.
    """
    return [
        GenerationResult(output_tokens=req.max_tokens, output_tokens_exact=False)
        for req in requests
    ]


class NnsightSteeringEngine(SteeringEngine):
    """Adapter over ``nnsight.LanguageModel`` trace-context steering."""

    name = "nnsight"
    capabilities = Capabilities(
        batching=True,  # pseudo-batch only; per-prompt token counts are inexact
        named_modules=False,
        multi_layer=False,
        multi_hook=False,
        capture=False,
    )

    def __init__(self) -> None:
        self._model: Any | None = None

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
        from nnsight import LanguageModel

        dtype = opts.pop("torch_dtype", torch.float16)
        device_map = opts.pop("device_map", "auto")
        self._model = LanguageModel(
            model_id, device_map=device_map, torch_dtype=dtype, **opts
        )

    def _resolve(self, steering: Any) -> tuple[int, tuple[float, ...]]:
        match steering:
            case SteeringSpec() as spec:
                _hook, layer, vec = resolve_single(spec)
                return layer, vec
            case NamedModuleRef():
                raise EngineError("nnsight does not support named steering modules")
            case None:
                raise EngineError("nnsight requires an inline steering spec")
            case other:  # pragma: no cover - guarded by types
                raise EngineError(f"unsupported steering: {type(other)!r}")

    def _generate_single(self, req: GenerationRequest) -> GenerationResult:
        import torch

        layer, vec = self._resolve(req.steering)
        tensor = torch.tensor(vec, dtype=torch.float16, device="cuda")
        with self._model.generate(
            req.prompt, max_new_tokens=req.max_tokens, do_sample=False
        ) as gen:
            with gen.invoke(req.prompt):
                self._model.model.layers[layer].output[0][:] += tensor
                output = self._model.output.save()
        prompt_len = len(self._model.tokenizer.encode(req.prompt))
        n_tokens = output.value.shape[1] - prompt_len
        return GenerationResult(output_tokens=n_tokens, output_tokens_exact=True)

    def _generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        import torch

        max_tokens = max(req.max_tokens for req in requests)
        with self._model.generate(max_new_tokens=max_tokens, do_sample=False) as gen:
            for req in requests:
                layer, vec = self._resolve(req.steering)
                tensor = torch.tensor(vec, dtype=torch.float16, device="cuda")
                with gen.invoke(req.prompt):
                    self._model.model.layers[layer].output[0][:] += tensor
        # nnsight's batched trace returns a combined output that cannot be split
        # back into per-prompt lengths, so we report the max_tokens placeholder
        # and flag it as inexact rather than pretending it is precise.
        return batch_placeholder_results(requests)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if self._model is None:
            raise RuntimeError("NnsightSteeringEngine.generate called before load()")
        if len(requests) == 1:
            return [self._generate_single(requests[0])]
        return self._generate_batch(requests)

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        self._model = None
        self._cleanup_gpu()

    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("nnsight")
        except PackageNotFoundError:
            return "unknown"
