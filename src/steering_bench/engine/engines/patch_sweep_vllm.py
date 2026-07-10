"""vLLM patch-sweep adapter for the :class:`PatchSweepEngine` seam.

Wraps :mod:`steering_bench.external.vllm_patch_sweep` (the verified HTTP driver)
without rewriting it: ``setup`` records the ``base_url`` and health-checks the
already-running ``--enable-patching`` server, and ``run_sweep`` issues one
``POST /v1/patch_sweep`` per call and maps the returned dict into a typed
:class:`PatchSweepResult` (carrying the vLLM-only ``noise_floor`` /
``auto_captured`` / ``skipped`` fields).

``httpx`` is imported LAZILY inside ``vllm_patch_sweep`` method bodies, so this
module imports without an HTTP stack or a running server. Availability is a
*reachable server*, not a Python package: the adapter is always importable but
``setup`` raises :class:`PatchSweepError` when no healthy server answers.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities
from steering_bench.engine.patch_sweep import (
    PatchSweepEngine,
    PatchSweepError,
    PatchSweepRequest,
    PatchSweepResult,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VllmPatchSweepEngine(PatchSweepEngine):
    """Patch-sweep adapter over the vLLM fork's ``POST /v1/patch_sweep`` endpoint."""

    name = "vllm"
    capabilities = Capabilities(patch_sweep=True)

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._timeout_s: float = 900.0

    def setup(
        self,
        model_id: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 900.0,
        **_opts: Any,
    ) -> None:
        from steering_bench.external.vllm_patch_sweep import server_healthy

        if not server_healthy(base_url):
            raise PatchSweepError(f"no healthy patch-sweep server at {base_url}")
        self._base_url = base_url
        self._timeout_s = timeout_s

    def run_sweep(self, request: PatchSweepRequest) -> PatchSweepResult:
        from steering_bench.external.vllm_patch_sweep import (
            PatchSweepServerError,
            run_patch_sweep,
        )

        if self._base_url is None:
            raise PatchSweepError("run_sweep called before setup()")
        try:
            raw = run_patch_sweep(
                self._base_url,
                request.clean,
                request.corrupt,
                request.answer,
                request.n_layers,
                self._timeout_s,
            )
        except PatchSweepServerError as e:
            # A rejected request (e.g. a server missing --patch-source-cache-bytes)
            # carries the server's own message + a launch-flag hint; surface it as
            # a PatchSweepError so the benchmark reports it cleanly.
            raise PatchSweepError(str(e)) from e
        return PatchSweepResult.from_vllm_dict(raw)

    def teardown(self) -> None:
        self._base_url = None

    # -- identity ------------------------------------------------------------

    def version(self) -> str:
        try:
            import vllm

            return vllm.__version__
        except ImportError:
            return "unknown"

    def commit(self) -> str | None:
        try:
            import vllm

            return getattr(vllm, "__commit__", None)
        except ImportError:
            return None
