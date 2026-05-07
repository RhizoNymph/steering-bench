#!/usr/bin/env python3
"""Steering latency with vs. without the capture-consumer plugin system.

The capture-consumer subsystem ships an always-loaded ``maybe_capture_residual``
gate inside ``apply_layer_steering``: when no manager is installed
(``--capture-consumers`` unset on the server, ``capture_consumers=None`` in
the LLM constructor) the call is a Python ``None`` check that ``torch.compile``
is supposed to constant-fold out of the compiled graph. When a manager *is*
installed the gate dispatches to ``torch.ops.vllm.capture_residual`` on every
``(layer, hook)`` invocation regardless of whether any per-request spec asked
for that position — the manager decides whether to materialize the row.

This benchmark measures whether enabling the plugin system actually adds
overhead to the steering hot path. For every steering mode (disabled,
enabled_idle, per_request_1, per_request_4) it sweeps three capture states:

  cap_off       no consumers, capture system inactive (cold path)
  cap_on_idle   one logging consumer registered globally on a single
                (post_mlp, layer L) point but no per-request spec — measures
                the manager-installed-but-mostly-idle worst case
  cap_on_active same logging consumer + per-request capture asking for
                last_prompt at (post_mlp, layer L) on every request

Output: same schema as ``bench_latency.py`` but tagged ``vllm.steering_with_capture``
plus the three new per-config columns. Each (mode, cap, batch_size) cell is
run in a fresh subprocess so vLLM's residual weight memory does not leak across
configs.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.capture_consumers.runner import run_in_subprocess
from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import (
    random_steering_vectors,
    random_steering_vectors_diverse,
)

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}

STEERING_MODES = ("disabled", "enabled_idle", "per_request_1", "per_request_4")
CAPTURE_MODES = ("cap_off", "cap_on_idle", "cap_on_active")


def _make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    return [" ".join(["hello"] * words_needed)] * num_prompts


def _build_capture_consumers(
    capture_mode: str,
    capture_layer: int,
) -> list[dict] | None:
    """Translate ``capture_mode`` into the LLM(capture_consumers=...) value."""
    if capture_mode == "cap_off":
        return None
    # Both cap_on_idle and cap_on_active register the same logging consumer.
    # The active variant simply adds a per-request capture spec on top.
    return [
        {
            "name": "logging",
            "params": {
                "hooks": {"post_mlp": [capture_layer]},
                "positions": "last_prompt",
                "level": "WARNING",
            },
        }
    ]


def _build_sampling_params(
    steering_mode: str,
    capture_mode: str,
    batch_size: int,
    max_tokens: int,
    hidden_size: int,
    num_layers: int,
    capture_layer: int,
):
    """Return the sampling-params list for one (steering, capture) cell."""
    from vllm import SamplingParams

    capture_field = None
    if capture_mode == "cap_on_active":
        capture_field = {
            "logging": {
                "hooks": {"post_mlp": [capture_layer]},
                "positions": "last_prompt",
            }
        }

    def _sp(steering_kwargs: dict | None = None):
        kw = dict(max_tokens=max_tokens, temperature=0.0)
        if steering_kwargs:
            kw.update(steering_kwargs)
        if capture_field is not None:
            kw["capture"] = capture_field
        return SamplingParams(**kw)

    if steering_mode in ("disabled", "enabled_idle"):
        return [_sp() for _ in range(batch_size)]

    if steering_mode == "per_request_1":
        vectors = random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        return [_sp({"steering_vectors": vectors}) for _ in range(batch_size)]

    if steering_mode == "per_request_4":
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_configs=4,
            hook_points=["post_mlp"],
            scale=0.1,
            base_seed=42,
        )
        return [
            _sp({"steering_vectors": diverse[i % 4]})
            for i in range(batch_size)
        ]

    raise ValueError(f"unknown steering mode: {steering_mode}")


def _run_cell(
    *,
    model: str,
    steering_mode: str,
    capture_mode: str,
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
    capture_layer: int,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict:
    """Run a single (steering, capture, batch_size) cell. Picklable for spawn."""
    from vllm import LLM

    enable_steering = steering_mode != "disabled"
    max_configs = 8 if steering_mode == "per_request_4" else 4

    capture_consumers = _build_capture_consumers(capture_mode, capture_layer)

    print(
        f"    [load] steering={enable_steering} max_configs={max_configs} "
        f"capture={capture_mode} prefix_cache={enable_prefix_caching}",
        flush=True,
    )
    llm = LLM(
        model=model,
        enable_steering=enable_steering,
        max_steering_configs=max_configs,
        capture_consumers=capture_consumers,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    prompts = _make_prompts(batch_size, prompt_len)
    sp_list = _build_sampling_params(
        steering_mode=steering_mode,
        capture_mode=capture_mode,
        batch_size=batch_size,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        num_layers=num_layers,
        capture_layer=capture_layer,
    )

    print(f"    [measure] warmup={warmup} iters={iters}", flush=True)
    try:
        for _ in range(warmup):
            llm.generate(prompts, sp_list)
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sp_list)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        stats = compute_stats(samples)
        return stats.to_dict()
    except torch.cuda.OutOfMemoryError:
        return {"error": "OOM", "samples_ms": []}
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Latency of steering modes with/without capture-consumers enabled "
            "(measures whether installing a capture manager adds overhead to "
            "the steering hot path even when no request asks for a capture)."
        )
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--steering-modes",
        default=",".join(STEERING_MODES),
        help=f"Comma-separated subset of {STEERING_MODES}",
    )
    parser.add_argument(
        "--capture-modes",
        default=",".join(CAPTURE_MODES),
        help=f"Comma-separated subset of {CAPTURE_MODES}",
    )
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=15,
        help="Layer index used by the capture consumer (clamped to num_layers-1).",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching to isolate per-step overhead.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help=(
            "Run all cells in this process (faster, but vLLM leaks weight memory "
            "across LLM constructions and may OOM after the first few cells)."
        ),
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=float,
        default=900.0,
        help="Per-cell subprocess wall-clock budget in seconds.",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    model_cfg = MODEL_CONFIGS.get(args.model)
    if model_cfg is None:
        print(
            f"Warning: unknown model {args.model}, defaulting to "
            "hidden_size=2560 num_layers=34"
        )
        model_cfg = {"hidden_size": 2560, "num_layers": 34}
    hidden_size = model_cfg["hidden_size"]
    num_layers = model_cfg["num_layers"]
    capture_layer = min(args.capture_layer, num_layers - 1)

    steering_modes = [m for m in args.steering_modes.split(",") if m]
    capture_modes = [m for m in args.capture_modes.split(",") if m]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    for m in steering_modes:
        if m not in STEERING_MODES:
            print(f"ERROR: unknown steering mode {m!r}")
            sys.exit(2)
    for m in capture_modes:
        if m not in CAPTURE_MODES:
            print(f"ERROR: unknown capture mode {m!r}")
            sys.exit(2)

    print(f"steering_with_capture: {args.model}")
    print(f"  steering_modes = {steering_modes}")
    print(f"  capture_modes  = {capture_modes}")
    print(f"  batch_sizes    = {batch_sizes}")
    print(f"  capture_layer  = {capture_layer}")
    print(f"  prefix_cache   = {not args.disable_prefix_cache}")
    print()

    # Baselines for overhead %: keyed by (steering_mode, batch_size); we
    # compare each capture_mode at fixed (steering, batch) against cap_off.
    # If cap_off isn't in the requested capture_modes the overhead column
    # stays None.
    baselines: dict[tuple[str, int], float] = {}
    all_results: list[dict] = []

    for steering_mode in steering_modes:
        for capture_mode in capture_modes:
            print(f"\n--- steering={steering_mode}  capture={capture_mode} ---")
            for batch_size in batch_sizes:
                cell_kwargs = dict(
                    model=args.model,
                    steering_mode=steering_mode,
                    capture_mode=capture_mode,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    max_tokens=args.max_tokens,
                    warmup=args.warmup,
                    iters=args.iters,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    capture_layer=capture_layer,
                    enable_prefix_caching=not args.disable_prefix_cache,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                )

                print(f"  batch_size={batch_size}")
                if args.no_subprocess:
                    result = _run_cell(**cell_kwargs)
                else:
                    result = run_in_subprocess(
                        target=_run_cell,
                        kwargs=cell_kwargs,
                        timeout_s=args.subprocess_timeout,
                    )

                if "error" in result and "mean_ms" not in result:
                    print(f"    {result['error']}")
                    overhead_pct = None
                else:
                    mean = result["mean_ms"]
                    p90 = result.get("p90_ms", float("nan"))
                    print(f"    mean={mean:.1f}ms p90={p90:.1f}ms")

                    if capture_mode == "cap_off":
                        baselines[(steering_mode, batch_size)] = mean

                    overhead_pct = None
                    base = baselines.get((steering_mode, batch_size))
                    if base is not None and base > 0:
                        overhead_pct = (mean - base) / base * 100.0
                        if capture_mode == "cap_off":
                            print(f"    (baseline for steering={steering_mode})")
                        else:
                            print(
                                f"    overhead vs cap_off: {overhead_pct:+.2f}%"
                            )

                params = {
                    "model": args.model,
                    "steering_mode": steering_mode,
                    "capture_mode": capture_mode,
                    "batch_size": batch_size,
                    "prompt_len": args.prompt_len,
                    "max_tokens": args.max_tokens,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "capture_layer": capture_layer,
                    "prefix_caching": not args.disable_prefix_cache,
                }
                results_dict: dict = {
                    "latency_ms": {
                        k: v for k, v in result.items() if k != "samples_ms"
                    },
                }
                if overhead_pct is not None:
                    results_dict["overhead_pct_vs_cap_off"] = overhead_pct

                write_result(
                    benchmark="vllm.steering_with_capture",
                    parameters=params,
                    results=results_dict,
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=result.get("samples_ms"),
                )
                all_results.append({
                    "steering_mode": steering_mode,
                    "capture_mode": capture_mode,
                    "batch_size": batch_size,
                    "mean_ms": result.get("mean_ms"),
                    "p90_ms": result.get("p90_ms"),
                    "overhead_pct": overhead_pct,
                    "error": result.get("error"),
                })

    # Summary
    print("\n" + "=" * 96)
    print(f"  Steering ⨯ Capture latency: {args.model}")
    print("=" * 96)
    header = (
        f"{'steering':<14} {'capture':<14} {'batch':>6} "
        f"{'mean_ms':>10} {'p90_ms':>10} {'overhead':>10}"
    )
    print(header)
    print("-" * 96)
    for r in all_results:
        if r["error"] and r["mean_ms"] is None:
            print(
                f"{r['steering_mode']:<14} {r['capture_mode']:<14} "
                f"{r['batch_size']:>6} {'ERROR':>10}"
            )
            continue
        overhead = (
            f"{r['overhead_pct']:+.2f}%"
            if r["overhead_pct"] is not None
            else "baseline"
        )
        print(
            f"{r['steering_mode']:<14} {r['capture_mode']:<14} "
            f"{r['batch_size']:>6} {r['mean_ms']:>10.1f} "
            f"{r['p90_ms']:>10.1f} {overhead:>10}"
        )
    print("=" * 96)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
