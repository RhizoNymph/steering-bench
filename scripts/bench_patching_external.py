#!/usr/bin/env python3
"""Cross-library activation-patching (causal tracing) sweep comparison.

Runs the identical denoising study — patch the clean run's residual into the
corrupt run at every (layer, position) cell, grade the answer token — through:

* ``tl_naive``   — TransformerLens, one forward per cell.
* ``tl_batched`` — TransformerLens, one forward per layer (position-batched,
  logits-chunked).
* ``vllm_sweep`` — vLLM's one-call ``POST /v1/patch_sweep`` against a running
  ``--enable-patching`` server (includes HTTP, server-side clean-run
  auto-capture, baselines, noise floor, and source cleanup).

The TransformerLens variants load the model in-process, so on a
memory-constrained GPU run them and the vLLM server measurement in separate
sessions (``--variants tl_batched,tl_naive`` with the server down, then
``--variants vllm_sweep`` with it up).

Beyond wall time, each result records the recovered-metric argmax cell so
cross-tool agreement doubles as a correctness check: both tools should find
the same causal site.

Examples::

    # vLLM side (server running with --enable-patching):
    uv run scripts/bench_patching_external.py \\
        --variants vllm_sweep --base-url http://localhost:8000/v1

    # TransformerLens side (server down; needs the transformerlens extra):
    uv run scripts/bench_patching_external.py --variants tl_batched,tl_naive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steering_bench.external.base import is_library_available
from steering_bench.output import write_result

_FILLER = (
    "In this geography quiz, we will answer questions about European "
    "countries, their capitals, rivers, and mountains. We will discuss "
    "history, culture, food, languages, and famous landmarks of each "
    "nation in turn, considering both ancient origins and modern life. "
)
PROMPT_PAIRS = {
    "short": (
        "The capital of France is",
        "The capital of Germany is",
    ),
    "long": (
        "In this geography quiz, we will answer questions about European "
        "countries, their capitals, rivers, and mountains. Read each "
        "question carefully and answer with exactly one word. Question one. "
        "The capital of France is",
        "In this geography quiz, we will answer questions about European "
        "countries, their capitals, rivers, and mountains. Read each "
        "question carefully and answer with exactly one word. Question one. "
        "The capital of Germany is",
    ),
    "xl": (
        _FILLER * 4 + "Question one. The capital of France is",
        _FILLER * 4 + "Question one. The capital of Germany is",
    ),
}
ANSWER = " Paris"


def _run_tl(args, variants: list[str]) -> list[dict[str, Any]]:
    from steering_bench.external.tl_patching import load_model, run_patch_sweep

    model = load_model(args.model, dtype=args.dtype)
    out = []
    for pname in args.prompts:
        clean, corrupt = PROMPT_PAIRS[pname]
        for variant in variants:
            r = run_patch_sweep(model, clean, corrupt, ANSWER, variant)
            r["prompt"] = pname
            print(
                f"  [{pname}/{r['variant']}] {r['cells']} cells in "
                f"{r['wall_s']}s ({r['cells_per_s']} cells/s), "
                f"argmax L{r['argmax']['layer']}@{r['argmax']['position']}"
            )
            out.append(r)
    return out


def _run_vllm(args) -> list[dict[str, Any]]:
    from steering_bench.external.vllm_patch_sweep import (
        run_patch_sweep,
        server_healthy,
    )

    if not server_healthy(args.base_url):
        print(f"  vllm_sweep: SKIPPED (no healthy server at {args.base_url})")
        return []
    out = []
    for pname in args.prompts:
        clean, corrupt = PROMPT_PAIRS[pname]
        for rep in range(args.reps):
            r = run_patch_sweep(
                args.base_url, clean, corrupt, ANSWER, args.num_layers
            )
            r["prompt"] = pname
            r["rep"] = rep
            print(
                f"  [{pname}/vllm_sweep rep{rep}] {r['cells']} cells in "
                f"{r['wall_s']}s ({r['cells_per_s']} cells/s), "
                f"argmax L{r['argmax']['layer']}@{r['argmax']['position']}"
            )
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument(
        "--variants",
        default="tl_batched,tl_naive,vllm_sweep",
        help="comma list of tl_naive, tl_batched, vllm_sweep",
    )
    ap.add_argument("--prompts", default="short,long,xl")
    ap.add_argument("--reps", type=int, default=3, help="vllm_sweep repetitions")
    ap.add_argument("--output-dir", default="results/patching")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.prompts = [p.strip() for p in args.prompts.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    results: list[dict[str, Any]] = []

    tl_variants = [v.removeprefix("tl_") for v in variants if v.startswith("tl_")]
    if tl_variants:
        if is_library_available("transformer_lens"):
            print(f"TransformerLens variants: {tl_variants}")
            results += _run_tl(args, tl_variants)
        else:
            print("tl_*: SKIPPED (transformer_lens not installed)")

    if "vllm_sweep" in variants:
        print("vLLM /v1/patch_sweep:")
        results += _run_vllm(args)

    if not results:
        print("nothing ran — check --variants / installs / server")
        sys.exit(1)

    # Cross-tool agreement: same causal site found by every variant that ran
    # a given prompt (position must match; the recovered plateau makes layer
    # ties within it expected).
    for pname in args.prompts:
        pos = {
            r["argmax"]["position"] for r in results if r["prompt"] == pname
        }
        if len(pos) > 1:
            print(f"WARNING [{pname}]: argmax positions disagree: {pos}")

    write_result(
        benchmark="external.patching_sweep",
        parameters={
            "model": args.model,
            "num_layers": args.num_layers,
            "dtype": args.dtype,
            "prompts": args.prompts,
            "variants": variants,
            "answer": ANSWER,
            "base_url": args.base_url,
            "reps": args.reps,
        },
        results={"runs": results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    print("\nSummary (cells/s):")
    for pname in args.prompts:
        row = [r for r in results if r["prompt"] == pname]
        if not row:
            continue
        best: dict[str, float] = {}
        for r in row:
            key = r["variant"]
            best[key] = max(best.get(key, 0.0), r["cells_per_s"])
        cells = row[0]["cells"]
        parts = "  ".join(f"{k}={v:g}" for k, v in sorted(best.items()))
        print(f"  {pname:>6} ({cells} cells): {parts}")


if __name__ == "__main__":
    main()
