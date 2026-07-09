"""TransformerLens steering engine adapter.

Ports the additive-hook logic from ``external/transformerlens_bench.py``:
steering is applied via a per-forward-pass hook callback on a HookedTransformer.
TransformerLens has no continuous batching and no per-element hooks, so it only
supports a single-hook, single-layer ``SteeringSpec`` and generates
sequentially. ``transformer_lens`` (and torch) are imported lazily inside
methods so this module is importable without them.
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

# hook_point name -> TransformerLens hook suffix within ``blocks.{layer}``.
_HOOK_SUFFIX: dict[str, str] = {
    "pre_attn": "hook_resid_pre",
    "post_attn": "hook_attn_out",
    "post_mlp": "hook_resid_post",
}
_DEFAULT_HOOK_SUFFIX = "hook_resid_post"


def _resolve_single(spec: SteeringSpec) -> tuple[str, int, tuple[float, ...]]:
    """Validate that ``spec`` is single-hook/single-layer and unpack it.

    Pure helper (no torch); raises ``EngineError`` on unsupported specs.
    """
    if spec.is_multi_hook():
        raise EngineError(
            f"transformerlens supports a single hook, got {spec.hooks()}"
        )
    hook = spec.hooks()[0]
    layers = spec.layers(hook)
    if len(layers) != 1:
        raise EngineError(
            f"transformerlens supports a single layer, got {layers} for hook {hook!r}"
        )
    layer = layers[0]
    return hook, layer, spec.vectors[hook][layer]


def _hook_name(hook: str, layer: int) -> str:
    """Map ``(hook, layer)`` to a TransformerLens hook name."""
    suffix = _HOOK_SUFFIX.get(hook, _DEFAULT_HOOK_SUFFIX)
    return f"blocks.{layer}.{suffix}"


class TransformerLensSteeringEngine(SteeringEngine):
    """Adapter over ``transformer_lens.HookedTransformer``."""

    name = "transformerlens"
    capabilities = Capabilities(
        batching=False,
        named_modules=False,
        multi_layer=False,
        multi_hook=False,
        capture=False,
    )

    def __init__(self) -> None:
        self._model: Any | None = None
        self._device: str = "cuda"

    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        # steering_config knobs (max_steering_configs / prefix caching) are
        # vLLM-fork concepts with no analog here; ignored (no-op).
        del steering_config
        from transformer_lens import HookedTransformer

        dtype = opts.pop("dtype", "float16")
        device = opts.pop("device", "cuda")
        self._device = str(device)
        self._model = HookedTransformer.from_pretrained(
            model_id, dtype=dtype, device=device, **opts
        )

    @staticmethod
    def _make_hook_fn(vector: Any):  # vector: torch.Tensor
        def hook_fn(activation: Any, hook: Any) -> Any:
            activation[:, :, :] += vector
            return activation

        return hook_fn

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        import torch

        if self._model is None:
            raise RuntimeError(
                "TransformerLensSteeringEngine.generate called before load()"
            )

        results: list[GenerationResult] = []
        for req in requests:
            match req.steering:
                case None:
                    fwd_hooks: list[tuple[str, Any]] = []
                case NamedModuleRef():
                    raise EngineError(
                        "transformerlens does not support named steering modules"
                    )
                case SteeringSpec() as spec:
                    hook, layer, vec = _resolve_single(spec)
                    tensor = torch.tensor(
                        vec, dtype=torch.float16, device=self._device
                    )
                    fwd_hooks = [(_hook_name(hook, layer), self._make_hook_fn(tensor))]
                case other:  # pragma: no cover - guarded by types
                    raise EngineError(f"unsupported steering: {type(other)!r}")

            tokens = self._model.to_tokens(req.prompt)
            input_len = tokens.shape[1]
            with torch.no_grad():
                output = self._model.generate(
                    tokens,
                    max_new_tokens=req.max_tokens,
                    do_sample=False,
                    fwd_hooks=fwd_hooks,
                )
            results.append(GenerationResult(output_tokens=output.shape[1] - input_len))

        return results

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        self._model = None
        self._cleanup_gpu()

    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("transformer-lens")
        except PackageNotFoundError:
            return "unknown"
