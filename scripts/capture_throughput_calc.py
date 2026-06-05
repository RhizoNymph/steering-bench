#!/usr/bin/env python3
"""Activation-capture bandwidth calculator.

Pure arithmetic companion to bench_capture_filesystem.py: given a model and
what you capture, how fast can you generate before the disk/network can't keep
up? Models the residual-stream capture this consumer does (one hidden_size
vector per captured layer per hook-point per token).

    bytes/token = layers_captured * hook_points * hidden_size * dtype_bytes

Rates use decimal MB (1 MB = 1e6 bytes) to match `dd` / the benchmark.

Default thresholds are the rates we actually measured on this cluster
(node1 -> node0 NFS, RTX 3090 / Ryzen 5950X / WD Red SA500 SATA SSD):
  30  MB/s  real-time single-stream, small per-step files (metadata-RPC bound)
  95  MB/s  large-file sequential over 1 GbE
  371 MB/s  large-file sequential over the 20G bond (~disk bound)
  398 MB/s  node0 SATA SSD write ceiling (O_DIRECT)

Examples:
  # Largest sustainable tok/s for residual capture, all layers, bf16:
  python scripts/capture_throughput_calc.py --model llama-8b

  # Two hook points/layer, only 4 layers, fp8, and check a 3000 tok/s target:
  python scripts/capture_throughput_calc.py --model llama-8b \
      --hook-points 2 --layers-captured 4 --dtype fp8 --tokens-per-sec 3000

  # Explicit dims instead of a preset:
  python scripts/capture_throughput_calc.py --layers 80 --hidden 8192
"""

from __future__ import annotations

import argparse

# (num_layers, hidden_size) for common models.
MODELS: dict[str, tuple[int, int]] = {
    "gpt2": (12, 768),
    "gpt2-xl": (48, 1600),
    "gemma-3-4b": (34, 2560),
    "llama-8b": (32, 4096),
    "mistral-7b": (32, 4096),
    "llama-13b": (40, 5120),
    "llama-70b": (80, 8192),
    "qwen2-72b": (80, 8192),
    "gpt3-175b": (96, 12288),
    "llama-405b": (126, 16384),
}

DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}

# (rate MB/s, label) — measured on this cluster; override with --rates.
DEFAULT_RATES: list[tuple[float, str]] = [
    (30.0, "real-time single-stream (small files, metadata-bound)"),
    (95.0, "large-file sequential, 1 GbE"),
    (371.0, "large-file sequential, 20G bond (~disk bound)"),
    (398.0, "node0 SATA SSD write ceiling"),
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Activation-capture bandwidth calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", choices=sorted(MODELS), help="Model preset.")
    p.add_argument("--layers", type=int, help="num_layers (overrides --model).")
    p.add_argument("--hidden", type=int, help="hidden_size (overrides --model).")
    p.add_argument(
        "--layers-captured", default="all",
        help="How many layers you capture: 'all' or an integer. Default all.",
    )
    p.add_argument(
        "--hook-points", type=int, default=1,
        help="Residual hook points captured per layer (e.g. resid_post=1, "
             "post-attn + post-mlp=2). Default 1.",
    )
    p.add_argument(
        "--dtype", choices=sorted(DTYPE_BYTES), default="bf16",
        help="Capture dtype. Default bf16 (2 bytes).",
    )
    p.add_argument(
        "--tokens-per-sec", type=float, default=None,
        help="If set, compute required MB/s for this generation rate and "
             "flag it against each threshold.",
    )
    p.add_argument(
        "--rates", default=None,
        help="Comma-separated MB/s thresholds to flag against "
             "(overrides the measured defaults).",
    )
    args = p.parse_args()

    # Resolve model dimensions.
    if args.layers and args.hidden:
        layers, hidden = args.layers, args.hidden
        name = f"custom(L={layers},H={hidden})"
    elif args.model:
        layers, hidden = MODELS[args.model]
        name = args.model
    else:
        p.error("provide --model, or both --layers and --hidden")

    if args.layers_captured == "all":
        layers_captured = layers
    else:
        layers_captured = int(args.layers_captured)
        if not 1 <= layers_captured <= layers:
            p.error(f"--layers-captured must be in 1..{layers} or 'all'")

    dtype_bytes = DTYPE_BYTES[args.dtype]
    hooks = args.hook_points
    if hooks < 1:
        p.error("--hook-points must be >= 1")

    bytes_per_token = layers_captured * hooks * hidden * dtype_bytes

    if args.rates:
        rates = [(float(x), "custom") for x in args.rates.split(",")]
    else:
        rates = DEFAULT_RATES

    print(f"Model:           {name}  (L={layers}, H={hidden})")
    print(f"Capturing:       {layers_captured} layer(s) x {hooks} hook-point(s)"
          f" x {args.dtype} ({dtype_bytes} B)")
    print(f"Bytes/token:     {bytes_per_token:,} B  "
          f"({bytes_per_token / 1024:.1f} KiB)")
    print()
    print("Max sustainable generation rate per disk/network threshold:")
    print(f"  {'rate (MB/s)':>12}  {'max tok/s':>10}   note")
    print("  " + "-" * 70)
    for rate, label in rates:
        max_tps = rate * 1e6 / bytes_per_token
        print(f"  {rate:>12.0f}  {max_tps:>10,.0f}   {label}")

    if args.tokens_per_sec is not None:
        req = bytes_per_token * args.tokens_per_sec / 1e6
        print()
        print(f"At {args.tokens_per_sec:,.0f} tok/s you produce "
              f"{req:,.1f} MB/s of activations:")
        for rate, label in rates:
            verdict = "OK" if req <= rate else f"OVER by {req / rate:.1f}x"
            print(f"  vs {rate:>6.0f} MB/s ({label}): {verdict}")

    print()
    print("Note: sustained average. Prefill emits a burst of "
          "prompt_len x bytes/token that buffers in host RAM.")


if __name__ == "__main__":
    main()
