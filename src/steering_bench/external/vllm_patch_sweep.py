"""vLLM one-call activation-patching sweep over HTTP.

Measures ``POST /v1/patch_sweep`` end-to-end against a running
patching-enabled server (``--enable-patching``). The single request covers
what the TransformerLens variants do in-process AND more: server-side
auto-capture of the clean run, both baselines, exact answer-token grading,
the batch-noise-floor rerun, and source cleanup — so the measured wall time
is a strict superset of the study.
"""

from __future__ import annotations

import time
import uuid
from typing import Any


def server_healthy(base_url: str, timeout: float = 3.0) -> bool:
    import httpx

    root = base_url.rstrip("/").removesuffix("/v1")
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
        f"{base_url.rstrip('/')}/patch_sweep", json=body, timeout=timeout_s
    )
    wall = time.perf_counter() - t0
    resp.raise_for_status()
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
