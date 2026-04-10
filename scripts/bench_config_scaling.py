#!/usr/bin/env python3
"""Ablation benchmark: max_steering_configs scaling.

1x6 sweep: max_steering_configs in [1, 2, 4, 8, 16, 32]
with fixed batch size to isolate table-size overhead.

Key question: does overhead scale with max_steering_configs, or stay flat?
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
from steering_bench.vectors import random_steering_vectors_diverse


def _gpu_used_mb(device: int = 0) -> float:
    """Return total GPU memory used on *device* in MB.

    Uses torch.cuda.mem_get_info which queries the driver directly,
    so it sees memory held by vLLM's EngineCore subprocess.
    torch.cuda.memory_allocated() only sees the current process's
    tensors and returns 0 for subprocess-allocated memory.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / (1024 * 1024)

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def run_config(
    model: str,
    max_configs: int,
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
) -> dict:
    """Run latency + memory measurement for a single max_steering_configs value."""
    from vllm import LLM, SamplingParams

    print(f"    Loading model (max_steering_configs={max_configs})...", flush=True)
    llm = LLM(
        model=model,
        enable_steering=True,
        max_steering_configs=max_configs,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    # Memory after model load — use mem_get_info to see subprocess allocations
    allocated_mb = _gpu_used_mb()

    prompts = make_prompts(batch_size, prompt_len)

    # Use as many distinct configs as the table allows (up to batch_size)
    actual_distinct = min(max_configs, batch_size)
    diverse = random_steering_vectors_diverse(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_configs=actual_distinct,
        hook_points=["post_mlp"],
        scale=0.1,
        base_seed=42,
    )
    sp_list = [
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            steering_vectors=diverse[i % actual_distinct],
        )
        for i in range(batch_size)
    ]

    # Warmup
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    # Measure
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    stats = compute_stats(samples)
    result = stats.to_dict()
    result["allocated_mb"] = allocated_mb
    result["actual_distinct"] = actual_distinct
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablation: max_steering_configs scaling")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/ablation/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--configs-sweep", default="1,2,4,8,16,32")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    config_counts = [int(x) for x in args.configs_sweep.split(",")]
    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    print(f"Config Scaling Ablation: {args.model}")
    print(f"Fixed batch_size={args.batch_size}")
    print(f"max_steering_configs sweep: {config_counts}")
    print()

    all_results = []
    baseline_mean = None

    for max_configs in config_counts:
        print(f"--- max_steering_configs={max_configs} ---")

        try:
            result = run_config(
                model=args.model,
                max_configs=max_configs,
                batch_size=args.batch_size,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
            )

            mean = result["mean_ms"]
            p90 = result["p90_ms"]
            mem = result["allocated_mb"]
            actual = result["actual_distinct"]

            if baseline_mean is None:
                baseline_mean = mean

            overhead_pct = (mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0.0

            print(
                f"    actual_distinct={actual} mean={mean:.1f}ms "
                f"p90={p90:.1f}ms mem={mem:.0f}MB overhead={overhead_pct:+.1f}%"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            result = {"error": "OOM"}
            overhead_pct = None

        params = {
            "model": args.model,
            "max_steering_configs": max_configs,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
        }
        results_out = {
            "latency_ms": {
                k: v
                for k, v in result.items()
                if k not in ("samples_ms", "allocated_mb", "actual_distinct", "error")
            },
            "allocated_mb": result.get("allocated_mb"),
            "actual_distinct": result.get("actual_distinct"),
        }
        if overhead_pct is not None:
            results_out["overhead_pct"] = overhead_pct

        write_result(
            benchmark="ablation.config_scaling",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=result.get("samples_ms"),
        )
        all_results.append({"max_configs": max_configs, "results": results_out, "raw": result})

    # Summary
    print(f"\n{'=' * 85}")
    print(f"  Config Scaling Ablation Summary: {args.model}")
    print(f"{'=' * 85}")
    print(
        f"{'max_configs':>12} {'distinct':>10} {'mean_ms':>10} {'p90_ms':>10} "
        f"{'memory_MB':>10} {'overhead':>10}"
    )
    print(f"{'-' * 85}")
    for r in all_results:
        raw = r["raw"]
        if "error" in raw:
            print(f"{r['max_configs']:>12} {'OOM':>10}")
            continue
        overhead = r["results"].get("overhead_pct")
        overhead_str = f"{overhead:+.1f}%" if overhead is not None else "baseline"
        print(
            f"{r['max_configs']:>12} {raw.get('actual_distinct', '?'):>10} "
            f"{raw['mean_ms']:>10.1f} {raw['p90_ms']:>10.1f} "
            f"{raw.get('allocated_mb', 0):>10.0f} {overhead_str:>10}"
        )
    print(f"{'=' * 85}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
