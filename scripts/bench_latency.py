#!/usr/bin/env python3
"""vLLM system benchmark: per-request latency with/without steering.

Measures end-to-end latency across steering modes and batch sizes.
This is the primary benchmark for the "X% overhead" headline number.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import (
    even_layer_subset,
    random_steering_vectors,
    random_steering_vectors_diverse,
)

# Named-module key used by the named_shared mode.  Single fixed name —
# the worker registers this once and every request references it via
# steering_module_ref.
NAMED_BENCH_MODULE = "bench_named_shared"

# Model constants (Gemma-3-4B-IT)
MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    """Generate dummy prompts of approximately the right token length."""
    # ~1.3 tokens per word on average
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def measure_latency(
    llm,
    prompts: list[str],
    sampling_params_list: list,
    warmup: int,
    iters: int,
) -> list[float]:
    """Run generate() iters times, return per-call wall-clock ms."""
    # Warmup
    for _ in range(warmup):
        llm.generate(prompts, sampling_params_list)

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    return samples


def _vectors_to_named_payload(vectors: dict) -> dict:
    """Convert random_steering_vectors output to register_steering_modules payload.

    Coerces numpy arrays to lists if present (the named-registration
    payload is stricter than the inline-vectors path).
    """
    import numpy as np

    coerced: dict = {}
    for hook, layer_dict in vectors.items():
        coerced[hook] = {}
        for layer_idx, entry in layer_dict.items():
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            coerced[hook][layer_idx] = entry
    return {"vectors": coerced}


def run_mode(
    model: str,
    mode: str,
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
    enable_prefix_caching: bool = True,
    num_hooks: int = 1,
    num_layers_steered: int | None = None,
) -> dict:
    """Run a single (mode, batch_size) configuration and return results.

    Recognized modes:

    - ``disabled`` — steering subsystem off; bare baseline.
    - ``enabled_idle`` — steering on but no per-request vectors.  Measures
      the fixed steering-on overhead in the absence of actual vectors.
    - ``per_request_1`` — same shared spec on every request via ``[sp]*N``.
      Auto-promote on `feat/steering` will lift this to a named module on
      second sight, so this mode reflects the *post-promote* steady state.
    - ``per_request_4`` — cycle four distinct specs across the batch.
    - ``inline_shared`` — alias for ``per_request_1`` (clearer name in the
      modes-matrix output).
    - ``inline_unique`` — every request gets a *fresh* spec (different seed).
      This is the research-style workload where auto-promote can never
      amortize via cache hits.
    - ``named_shared`` — pre-register a single module via
      ``register_steering_modules`` and have every request reference it via
      ``steering_module_ref``.  This is the floor for the spec-reuse case.

    Sweeps ``num_hooks`` (1, 2, or 3 — picked from
    ``[\"post_mlp\", \"post_attn\", \"pre_attn\"]``) and
    ``num_layers_steered`` (subset of layers via
    :func:`even_layer_subset`) so the matrix runner can attribute cost
    to "how much" steering is happening.
    """
    from vllm import LLM, SamplingParams

    prompts = make_prompts(batch_size, prompt_len)

    enable_steering = mode != "disabled"
    if mode == "per_request_4":
        # 4 inline first-sight slots + 4 auto-promoted named slots +
        # headroom for in-flight transitions.
        max_configs = 16
    elif mode == "inline_unique":
        # Every request submits a fresh spec; the worker table needs to
        # hold at least the per-iter unique count, plus headroom across
        # warmup/timed iters and auto-promote LRU thrash.
        max_configs = max(64, batch_size * 2)
    else:
        max_configs = 4

    # Pick the hook points and layer subset shared by every steering mode
    # in this run.
    all_hooks = ["post_mlp", "post_attn", "pre_attn"]
    if num_hooks < 1 or num_hooks > len(all_hooks):
        raise ValueError(
            f"num_hooks must be in [1, {len(all_hooks)}], got {num_hooks}"
        )
    active_hooks = all_hooks[:num_hooks]
    layer_subset = (
        None if num_layers_steered is None
        else even_layer_subset(num_layers, num_layers_steered)
    )

    print(
        f"    Loading model (enable_steering={enable_steering}, "
        f"max_configs={max_configs}, prefix_cache={enable_prefix_caching}, "
        f"hooks={active_hooks}, layers_steered={len(layer_subset) if layer_subset else num_layers})...",
        flush=True,
    )
    llm = LLM(
        model=model,
        enable_steering=enable_steering,
        max_steering_configs=max_configs,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    # Build sampling params based on mode
    if mode == "disabled" or mode == "enabled_idle":
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        sp_list = [sp] * batch_size

    elif mode in ("per_request_1", "inline_shared"):
        vectors = random_steering_vectors(
            hidden_size=hidden_size, num_layers=num_layers,
            hook_points=active_hooks, scale=0.1, seed=42,
            layer_subset=layer_subset,
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_vectors=vectors,
        )
        sp_list = [sp] * batch_size

    elif mode == "per_request_4":
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size, num_layers=num_layers,
            num_configs=4, hook_points=active_hooks, scale=0.1, base_seed=42,
            layer_subset=layer_subset,
        )
        sp_list = []
        for i in range(batch_size):
            sp = SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                steering_vectors=diverse[i % 4],
            )
            sp_list.append(sp)

    elif mode == "inline_unique":
        # One unique spec per request — defeats auto-promote dedup.
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size, num_layers=num_layers,
            num_configs=batch_size, hook_points=active_hooks, scale=0.1,
            base_seed=42, layer_subset=layer_subset,
        )
        sp_list = [
            SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                steering_vectors=diverse[i],
            )
            for i in range(batch_size)
        ]

    elif mode == "named_shared":
        # Pre-register one module on every worker via collective_rpc, then
        # have each request reference it via (name, scale).  This is the
        # ideal-case "shared spec" floor — only 16 bytes per request hits
        # the wire.
        vectors = random_steering_vectors(
            hidden_size=hidden_size, num_layers=num_layers,
            hook_points=active_hooks, scale=0.1, seed=42,
            layer_subset=layer_subset,
        )
        llm.llm_engine.collective_rpc(
            "register_steering_modules",
            kwargs={
                "modules": {NAMED_BENCH_MODULE: _vectors_to_named_payload(vectors)},
                "replace": True,
            },
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_module_ref=(NAMED_BENCH_MODULE, 1.0),
        )
        sp_list = [sp] * batch_size

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"    Measuring (warmup={warmup}, iters={iters})...", flush=True)
    try:
        samples = measure_latency(llm, prompts, sp_list, warmup, iters)
        stats = compute_stats(samples)
        result = stats.to_dict()
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM at batch_size={batch_size}!")
        result = {"error": "OOM", "samples_ms": []}
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM latency with/without steering")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching. Use to isolate its effect on "
             "per_request steering overhead.",
    )
    parser.add_argument(
        "--modes",
        default=(
            "disabled,enabled_idle,inline_shared,inline_unique,named_shared,"
            "per_request_4"
        ),
        help="Comma-separated list of modes to run.  See run_mode docstring "
             "for the catalog.",
    )
    parser.add_argument(
        "--num-hooks",
        type=int,
        default=1,
        help="Number of hook points to populate per request (1..3, picked "
             "from [post_mlp, post_attn, pre_attn]).",
    )
    parser.add_argument(
        "--num-layers-steered",
        type=int,
        default=None,
        help="Number of layers to steer (defaults to all model layers).  "
             "Selected as evenly-spaced indices via "
             "steering_bench.vectors.even_layer_subset.",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    config = MODEL_CONFIGS.get(args.model)
    if config is None:
        print(f"Warning: unknown model {args.model}, using default hidden_size=2560, num_layers=34")
        config = {"hidden_size": 2560, "num_layers": 34}

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    total = len(modes) * len(batch_sizes)
    print(f"Latency benchmark: {args.model}")
    print(f"Modes: {modes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total configs: {total}")
    print()

    all_results = []
    baseline_latency: dict[int, float] = {}  # batch_size -> disabled mean_ms

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        for batch_size in batch_sizes:
            print(f"  batch_size={batch_size}")

            result = run_mode(
                model=args.model,
                mode=mode,
                batch_size=batch_size,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=hidden_size,
                num_layers=num_layers,
                enable_prefix_caching=not args.disable_prefix_cache,
                num_hooks=args.num_hooks,
                num_layers_steered=args.num_layers_steered,
            )

            if "error" not in result:
                mean = result["mean_ms"]
                p90 = result["p90_ms"]
                print(f"    mean={mean:.1f}ms p90={p90:.1f}ms")

                if mode == "disabled":
                    baseline_latency[batch_size] = mean

                overhead_pct = None
                if batch_size in baseline_latency and baseline_latency[batch_size] > 0:
                    overhead_pct = (
                        (mean - baseline_latency[batch_size])
                        / baseline_latency[batch_size]
                        * 100
                    )
                    print(f"    overhead vs disabled: {overhead_pct:+.1f}%")
            else:
                print(f"    {result['error']}")
                overhead_pct = None

            params = {
                "model": args.model,
                "mode": mode,
                "batch_size": batch_size,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "prefix_caching": not args.disable_prefix_cache,
                "num_hooks": args.num_hooks,
                "num_layers_steered": (
                    args.num_layers_steered
                    if args.num_layers_steered is not None else num_layers
                ),
            }
            results_dict = {
                "latency_ms": {k: v for k, v in result.items() if k != "samples_ms"},
            }
            if overhead_pct is not None:
                results_dict["overhead_pct"] = overhead_pct

            write_result(
                benchmark="vllm.latency",
                parameters=params,
                results=results_dict,
                output_dir=args.output_dir,
                tag=args.tag,
                raw_samples_ms=result.get("samples_ms"),
            )
            all_results.append({
                "mode": mode,
                "batch_size": batch_size,
                "results": results_dict,
            })

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"  Latency Benchmark Summary: {args.model}")
    print(f"{'=' * 90}")
    print(f"{'mode':<18} {'batch':>6} {'mean_ms':>10} {'median_ms':>10} {'p90_ms':>10} {'overhead':>10}")
    print(f"{'-' * 90}")
    for r in all_results:
        lat = r["results"].get("latency_ms", {})
        if "error" in lat:
            print(f"{r['mode']:<18} {r['batch_size']:>6} {'OOM':>10}")
            continue
        overhead = r["results"].get("overhead_pct")
        overhead_str = f"{overhead:+.1f}%" if overhead is not None else "baseline"
        print(
            f"{r['mode']:<18} {r['batch_size']:>6} "
            f"{lat.get('mean_ms', 0):>10.1f} {lat.get('median_ms', 0):>10.1f} "
            f"{lat.get('p90_ms', 0):>10.1f} {overhead_str:>10}"
        )
    print(f"{'=' * 90}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
