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
from steering_bench.vectors import random_steering_vectors, random_steering_vectors_diverse

# Model constants (Gemma-3-4B-IT)
MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
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
) -> dict:
    """Run a single (mode, batch_size) configuration and return results."""
    from vllm import LLM, SamplingParams

    prompts = make_prompts(batch_size, prompt_len)

    enable_steering = mode != "disabled"
    max_configs = 8 if mode == "per_request_4" else 4

    print(f"    Loading model (enable_steering={enable_steering}, max_configs={max_configs})...",
          flush=True)
    llm = LLM(
        model=model,
        enable_steering=enable_steering,
        max_steering_configs=max_configs,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    # Build sampling params based on mode
    if mode == "disabled" or mode == "enabled_idle":
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        sp_list = [sp] * batch_size

    elif mode == "per_request_1":
        vectors = random_steering_vectors(
            hidden_size=hidden_size, num_layers=num_layers,
            hook_points=["post_mlp"], scale=0.1, seed=42,
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_vectors=vectors,
        )
        sp_list = [sp] * batch_size

    elif mode == "per_request_4":
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size, num_layers=num_layers,
            num_configs=4, hook_points=["post_mlp"], scale=0.1, base_seed=42,
        )
        sp_list = []
        for i in range(batch_size):
            sp = SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                steering_vectors=diverse[i % 4],
            )
            sp_list.append(sp)
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
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    modes = ["disabled", "enabled_idle", "per_request_1", "per_request_4"]

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
