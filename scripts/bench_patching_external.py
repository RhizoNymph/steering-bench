#!/usr/bin/env python3
"""Cross-library activation-patching (causal tracing) sweep comparison.

Thin driver over the :class:`~steering_bench.engine.patch_sweep.PatchSweepEngine`
seam: it parses the legacy flags and hands off to
:class:`~steering_bench.harness.benchmarks.patch_sweep.PatchSweepBenchmark`, which
owns adapter discovery, the sweep loop, per-engine result writing, the cross-tool
argmax-agreement check, and the cells/s summary. Equivalent to::

    python -m steering_bench run patch-sweep --variants ... --prompts ...

Runs the identical denoising study -- patch the clean run's residual into the
corrupt run at every (layer, position) cell, grade the answer token -- through:

* ``tl_naive``   -- TransformerLens, one forward per cell.
* ``tl_batched`` -- TransformerLens, one forward per layer (position-batched).
* ``vllm_sweep`` -- vLLM's one-call ``POST /v1/patch_sweep`` against a running
  ``--enable-patching`` server.

The TransformerLens variants load the model in-process, so on a
memory-constrained GPU run them and the vLLM server measurement in separate
sessions (``--variants tl_batched,tl_naive`` with the server down, then
``--variants vllm_sweep`` with it up).

Examples::

    # vLLM side (server running with --enable-patching):
    uv run scripts/bench_patching_external.py \\
        --variants vllm_sweep --base-url http://localhost:8000/v1

    # TransformerLens side (server down; needs the transformerlens extra):
    uv run scripts/bench_patching_external.py --variants tl_batched,tl_naive
"""

from __future__ import annotations

import argparse

from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.harness.benchmarks.patch_sweep import PatchSweepBenchmark


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tag", default="")
    ap.add_argument("--output-dir", default="results/patching")
    PatchSweepBenchmark.add_args(ap)
    args = ap.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        warmup=0,
        iters=0,
        max_tokens=1,
        layer=0,
        hook="post_block",
        output_dir=args.output_dir,
        tag=args.tag,
    )
    options = PatchSweepBenchmark.options_from_args(args)
    PatchSweepBenchmark(config, **options).run()


if __name__ == "__main__":
    main()
