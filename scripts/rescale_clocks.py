#!/usr/bin/env python3
"""Rescale benchmark JSON throughput/latency to a target GPU clock.

Walks ``--input-dir`` recursively, scales every ``*_tps`` field by
``target_mhz / env.gpu_clock_current_mhz`` and every ``*_ms`` field by
the reciprocal, then writes the result to ``--output-dir`` preserving
the relative layout. Records without an environment clock are copied
unchanged.

Stamps each rewritten record with
``environment.gpu_clock_rescaled_from_mhz`` and
``environment.gpu_clock_rescaled_to_mhz`` so downstream consumers can
see what happened.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


_TPS_SUFFIX = "_tps"
_MS_SUFFIX = "_ms"
# Keys whose numeric values are sample counts / ratios — not clock-scaled.
_INVARIANT_KEYS = {
    "n",
    "avg_output_tokens_total",
    "latency_overhead_pct",
    "throughput_loss_pct",
    "num_ok",
    "num_err",
    "num_prompts",
    "total_output_tokens",
}


def _scale_block(d: dict, scale: float) -> None:
    """Mutate ``d`` in place, scaling tps fields by ``scale`` and ms
    fields by ``1/scale``. Recurses into nested dicts."""
    for k, v in list(d.items()):
        if isinstance(v, dict):
            _scale_block(v, scale)
            continue
        if k in _INVARIANT_KEYS:
            continue
        if not isinstance(v, (int, float)):
            continue
        if isinstance(v, bool):
            continue
        if k.endswith(_TPS_SUFFIX):
            d[k] = v * scale
        elif k.endswith(_MS_SUFFIX):
            d[k] = v / scale


def rescale_file(src: Path, dst: Path, target_mhz: float) -> str:
    with src.open() as f:
        data = json.load(f)
    env = data.get("environment", {})
    current = env.get("gpu_clock_current_mhz")
    if not isinstance(current, (int, float)) or current <= 0:
        # No clock metadata — copy unchanged.
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return "copied (no clock)"

    scale = target_mhz / float(current)
    results = data.get("results")
    if isinstance(results, dict):
        _scale_block(results, scale)
    env["gpu_clock_rescaled_from_mhz"] = current
    env["gpu_clock_rescaled_to_mhz"] = target_mhz
    env["gpu_clock_current_mhz"] = target_mhz
    data["environment"] = env

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(data, f, indent=2)
    return f"rescaled ×{scale:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--target-mhz",
        type=float,
        required=True,
        help="Target clock (e.g. 1905 to match old gpu2 postfix-hash runs)",
    )
    args = ap.parse_args()

    src_root = Path(args.input_dir).resolve()
    dst_root = Path(args.output_dir).resolve()
    if not src_root.exists():
        raise SystemExit(f"input dir does not exist: {src_root}")

    count = 0
    for path in sorted(src_root.rglob("*.json")):
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        status = rescale_file(path, dst, args.target_mhz)
        print(f"  {rel} — {status}")
        count += 1
    print(f"\nProcessed {count} files → {dst_root}")


if __name__ == "__main__":
    main()
