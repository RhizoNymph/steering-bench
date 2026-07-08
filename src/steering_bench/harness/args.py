"""Shared CLI arguments for benchmark scripts and the harness CLI.

``add_common_args(parser)`` adds the flags every benchmark shares. It is
composable: a script (or the CLI ``run`` subcommand) adds its own extra flags on
top. Defaults match the legacy scripts so behavior is unchanged.
"""

from __future__ import annotations

import argparse

from steering_bench.engine.registry import ENGINE_REGISTRY

DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_OUTPUT_DIR = "results/"
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_MAX_TOKENS = 128
DEFAULT_LAYER = 8
DEFAULT_HOOK = "post_block"
DEFAULT_ENGINE = "vllm"


def engine_names() -> list[str]:
    """Engine names known to the registry (choices for ``--engine``)."""
    return [entry.name for entry in ENGINE_REGISTRY]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared across benchmark scripts to ``parser``."""
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write JSON results into.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Unmeasured warmup iters."
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS, help="Measured iterations."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Tokens to generate per request.",
    )
    parser.add_argument(
        "--layer", type=int, default=DEFAULT_LAYER, help="Layer index to steer."
    )
    parser.add_argument(
        "--hook", default=DEFAULT_HOOK, help="Hook point to steer (e.g. post_block)."
    )
    parser.add_argument("--tag", default="", help="Optional tag recorded in results.")
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=engine_names(),
        help="Engine adapter to benchmark.",
    )
