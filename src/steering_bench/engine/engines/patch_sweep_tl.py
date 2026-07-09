"""TransformerLens patch-sweep adapter for the :class:`PatchSweepEngine` seam.

Wraps :mod:`steering_bench.external.tl_patching` (the verified causal-tracing
logic) without rewriting it: ``setup`` loads a ``HookedTransformer`` in-process
via ``load_model``, and ``run_sweep`` calls ``run_patch_sweep`` for the requested
``naive`` / ``batched`` variant and maps the returned dict into a typed
:class:`PatchSweepResult`.

``transformer_lens`` / ``torch`` are imported LAZILY inside method bodies (they
live in ``tl_patching``), so this module imports without them.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities, cleanup_gpu
from steering_bench.engine.patch_sweep import (
    PatchSweepEngine,
    PatchSweepError,
    PatchSweepRequest,
    PatchSweepResult,
)

TL_VARIANTS: tuple[str, ...] = ("naive", "batched")


class TLPatchSweepEngine(PatchSweepEngine):
    """Patch-sweep adapter over TransformerLens (in-process ``HookedTransformer``)."""

    name = "tl"
    capabilities = Capabilities(patch_sweep=True)

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str | None = None

    def setup(self, model_id: str, *, dtype: str = "bfloat16", **_opts: Any) -> None:
        from steering_bench.external.tl_patching import load_model

        self._model = load_model(model_id, dtype=dtype)
        self._model_id = model_id

    def run_sweep(self, request: PatchSweepRequest) -> PatchSweepResult:
        from steering_bench.external.tl_patching import run_patch_sweep

        if self._model is None:
            raise PatchSweepError("run_sweep called before setup()")
        variant = request.variant
        if variant not in TL_VARIANTS:
            raise PatchSweepError(
                f"unknown TransformerLens variant {variant!r}; "
                f"known: {', '.join(TL_VARIANTS)}"
            )
        raw = run_patch_sweep(
            self._model,
            request.clean,
            request.corrupt,
            request.answer,
            variant,
            request.logits_chunk_budget,
        )
        return PatchSweepResult.from_tl_dict(raw)

    def teardown(self) -> None:
        self._model = None
        cleanup_gpu()

    # -- identity ------------------------------------------------------------

    def version(self) -> str:
        try:
            import transformer_lens

            return getattr(transformer_lens, "__version__", "unknown")
        except ImportError:
            return "unknown"
