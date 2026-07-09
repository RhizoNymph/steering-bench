#!/usr/bin/env python3
"""Deprecated shim: cross-library steering comparison.

The per-library ``external/*`` adapters this script used to drive were ported to
the ``SteeringEngine`` seam and retired.  Its successor is the seam-native
``external-comparison`` harness benchmark::

    python -m steering_bench run external-comparison --engine <engine>

This shim maps the old flags onto that command and forwards to it, so existing
invocations keep working.  ``--batch-size 1`` is the old Tier-1 single-request
comparison; ``--batch-size N`` is the old Tier-2 batched comparison.  Flags with
no seam analogue (``--libraries``, ``--skip-tier1/2``) are accepted and ignored
with a note.
"""

from __future__ import annotations

import argparse
import sys

from steering_bench.__main__ import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="[deprecated] forwards to `run external-comparison`"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="results/external/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--hook", default="post_block")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--tag", default="")
    # Accepted for compatibility, no seam analogue:
    parser.add_argument("--libraries", default="all")
    parser.add_argument("--skip-tier1", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    args = parser.parse_args()

    print(
        "DEPRECATED: scripts/bench_external.py forwards to "
        "`python -m steering_bench run external-comparison`.",
        file=sys.stderr,
    )
    for name, val in (
        ("--libraries", args.libraries != "all"),
        ("--skip-tier1", args.skip_tier1),
        ("--skip-tier2", args.skip_tier2),
    ):
        if val:
            print(f"  note: {name} has no engine-seam analogue; ignored.", file=sys.stderr)

    forwarded = [
        "run",
        "external-comparison",
        "--model", args.model,
        "--output-dir", args.output_dir,
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--max-tokens", str(args.max_tokens),
        "--batch-size", str(args.batch_size),
        "--layer", str(args.layer),
        "--hook", args.hook,
        "--prompt-len", str(args.prompt_len),
        "--tag", args.tag,
    ]
    return cli_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
