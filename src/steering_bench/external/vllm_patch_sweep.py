"""vLLM one-call activation-patching sweep over HTTP.

Measures ``POST /v1/patch_sweep`` end-to-end against a running server. The
single request covers what the TransformerLens variants do in-process AND
more: server-side auto-capture of the clean run, both baselines, exact
answer-token grading, the batch-noise-floor rerun, and source cleanup — so
the measured wall time is a strict superset of the study.

Server requirements
-------------------
The target server must be launched with::

    --enable-patching                    # mount the /v1/patch_sweep route
    --capture-consumers patch_source     # the consumer that stores clean-run rows

The clean-run source store **auto-sizes from the model** as of the fork's
source-store auto-default (RhizoNymph/vllm#262), so ``--patch-source-cache-bytes``
is optional — pass it only to override the auto budget, or ``0`` to disable the
store. On fork builds predating that fix the store defaults to *disabled*, so
there ``--patch-source-cache-bytes <N>`` (e.g. 2000000000) is also required.

The one-call auto-capture path (this client sends ``clean_prompt`` + a fresh
``source_run``) taps the clean run through the ``patch_source`` consumer into
that store. If the store is absent, captured rows are silently dropped and the
sweep 400s with ``patch source not found``; :func:`run_patch_sweep` surfaces
that server message (with a flag hint) rather than a bare HTTP error, so a
misconfigured server is obvious.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

_SERVER_FLAG_HINT = (
    "Launch the server with: --enable-patching --capture-consumers patch_source "
    "(the source store auto-sizes as of the fork's source-store auto-default; on "
    "older builds also pass --patch-source-cache-bytes <N>, e.g. 2000000000). A "
    "missing source store silently drops captured rows, yielding this 'patch "
    "source not found' 400."
)


class PatchSweepServerError(RuntimeError):
    """The patch-sweep server rejected the request (non-2xx), with its message."""


def _server_error_detail(resp: Any) -> str:
    """Extract the server's ``{"error": ...}`` message from a failed response.

    Appends the launch-flag hint when the message names the missing-source /
    patching-disabled conditions, so the operator sees the fix inline instead
    of an opaque ``HTTP 400``.
    """
    try:
        detail = resp.json().get("error") or resp.text
    except (ValueError, AttributeError):
        detail = getattr(resp, "text", "") or "<no response body>"
    detail = str(detail)
    lowered = detail.lower()
    if "patch source not found" in lowered or "enable-patching" in lowered:
        detail = f"{detail}\n{_SERVER_FLAG_HINT}"
    return f"patch_sweep HTTP {resp.status_code}: {detail}"


def normalize_base_url(base_url: str) -> str:
    """Normalize a server base URL to its OpenAI-compatible ``/v1`` root.

    Accepts either a bare host (``http://host:port``) or an already-``/v1``
    -suffixed URL and returns the ``/v1`` form, so patch-sweep callers pass the
    same bare base URL the serving path accepts (which auto-appends ``/v1``).
    Idempotent: a URL already ending in ``/v1`` is returned unchanged apart from
    any trailing slash.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def server_healthy(base_url: str, timeout: float = 3.0) -> bool:
    import httpx

    root = normalize_base_url(base_url).removesuffix("/v1")
    try:
        return httpx.get(f"{root}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def run_patch_sweep(
    base_url: str,
    clean: str,
    corrupt: str,
    answer: str,
    n_layers: int,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """One ``/v1/patch_sweep`` call: all layers x all prompt positions.

    A fresh run name per call keeps the measurement honest — every rep pays
    the full one-call cost including the clean-run auto-capture.
    """
    import httpx

    body = {
        "prompt": corrupt,
        "clean_prompt": clean,
        "source_run": f"bench-{uuid.uuid4().hex[:12]}",
        "layers": {"start": 0, "stop": n_layers},
        "positions": "all_prompt",
        "metric": "recovered",
        "answer_token": answer,
    }
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{normalize_base_url(base_url)}/patch_sweep", json=body, timeout=timeout_s
    )
    wall = time.perf_counter() - t0
    if resp.status_code >= 400:
        raise PatchSweepServerError(_server_error_detail(resp))
    data = resp.json()

    cells = sum(1 for row in data["grid"] for v in row if v is not None)
    best = data.get("argmax") or {}
    return {
        "variant": "vllm_sweep",
        "cells": cells,
        "n_layers": len(data["layers"]),
        "n_positions": len(data["positions"]),
        "wall_s": round(wall, 3),
        "cells_per_s": round(cells / wall, 1),
        "clean_logprob": data.get("clean"),
        "corrupt_logprob": data.get("corrupt"),
        "noise_floor": data.get("noise_floor"),
        "auto_captured": data.get("auto_captured"),
        "skipped": len(data.get("skipped") or []),
        "argmax": {
            "layer": best.get("layer"),
            "position": best.get("position"),
            "recovered": round(best["value"], 4) if best.get("value") else None,
        },
    }
