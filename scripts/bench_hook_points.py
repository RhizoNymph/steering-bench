#!/usr/bin/env python3
"""Ablation benchmark: active hook point count.

1x3 sweep: 1 hook (post_mlp), 2 hooks (post_attn+post_mlp), 3 hooks (all).

Key question: does the number of non-zero hook points affect latency?
The steering op runs at all 3 hook points regardless — even zero-path calls
execute gather+add. This measures whether non-zero hook count matters.
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
from steering_bench.vectors import random_steering_vectors

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}

HOOK_CONFIGS = [
    {
        "name": "1_hook",
        "active": ["post_mlp"],
        "description": "post_mlp only",
    },
    {
        "name": "2_hooks",
        "active": ["post_attn", "post_mlp"],
        "description": "post_attn + post_mlp",
    },
    {
        "name": "3_hooks",
        "active": ["pre_attn", "post_attn", "post_mlp"],
        "description": "all three",
    },
]


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def run_hook_config(
    model: str,
    active_hooks: list[str],
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
) -> dict:
    """Run latency measurement for a specific set of active hooks."""
    from vllm import LLM, SamplingParams

    hook_label = "+".join(active_hooks)
    print(f"    Loading model (hooks={hook_label})...", flush=True)
    llm = LLM(
        model=model,
        enable_steering=True,
        max_steering_configs=4,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompts = make_prompts(batch_size, prompt_len)

    # Generate vectors only for active hooks — others stay zero
    vectors = random_steering_vectors(
        hidden_size=hidden_size,
        num_layers=num_layers,
        hook_points=active_hooks,
        scale=0.1,
        seed=42,
    )
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        steering_vectors=vectors,
    )
    sp_list = [sp] * batch_size

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

    return compute_stats(samples).to_dict()


def main():
    parser = argparse.ArgumentParser(description="Ablation: active hook point count")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/ablation/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    total = len(HOOK_CONFIGS) * len(batch_sizes)
    print(f"Hook Points Ablation: {args.model}")
    print(f"Hook configs: {[h['name'] for h in HOOK_CONFIGS]}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total configurations: {total}")
    print()

    all_results = []
    # baseline_latency[batch_size] = mean_ms for 1_hook config
    baseline_latency: dict[int, float] = {}

    for hook_cfg in HOOK_CONFIGS:
        print(f"\n--- {hook_cfg['name']}: {hook_cfg['description']} ---")
        for batch_size in batch_sizes:
            print(f"  batch_size={batch_size}")

            try:
                stats = run_hook_config(
                    model=args.model,
                    active_hooks=hook_cfg["active"],
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    max_tokens=args.max_tokens,
                    warmup=args.warmup,
                    iters=args.iters,
                    hidden_size=model_config["hidden_size"],
                    num_layers=model_config["num_layers"],
                )

                mean = stats["mean_ms"]
                p90 = stats["p90_ms"]

                if hook_cfg["name"] == "1_hook":
                    baseline_latency[batch_size] = mean

                overhead_pct = None
                if batch_size in baseline_latency and baseline_latency[batch_size] > 0:
                    overhead_pct = (
                        (mean - baseline_latency[batch_size])
                        / baseline_latency[batch_size]
                        * 100
                    )

                overhead_str = f"{overhead_pct:+.1f}%" if overhead_pct is not None else "baseline"
                print(f"    mean={mean:.1f}ms p90={p90:.1f}ms overhead={overhead_str}")

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM!")
                stats = {"error": "OOM"}
                overhead_pct = None

            params = {
                "model": args.model,
                "hook_config": hook_cfg["name"],
                "active_hooks": hook_cfg["active"],
                "num_active_hooks": len(hook_cfg["active"]),
                "batch_size": batch_size,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
            }
            results_out = {
                "latency_ms": {k: v for k, v in stats.items() if k != "samples_ms"},
            }
            if overhead_pct is not None:
                results_out["overhead_pct"] = overhead_pct

            write_result(
                benchmark="ablation.hook_points",
                parameters=params,
                results=results_out,
                output_dir=args.output_dir,
                tag=args.tag,
                raw_samples_ms=stats.get("samples_ms"),
            )
            all_results.append({
                "hook_config": hook_cfg["name"],
                "batch_size": batch_size,
                "results": results_out,
                "stats": stats,
            })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  Hook Points Ablation Summary: {args.model}")
    print(f"{'=' * 80}")
    print(f"{'hooks':<12} {'batch':>6} {'mean_ms':>10} {'p90_ms':>10} {'overhead':>10}")
    print(f"{'-' * 80}")
    for r in all_results:
        s = r["stats"]
        if "error" in s:
            print(f"{r['hook_config']:<12} {r['batch_size']:>6} {'OOM':>10}")
            continue
        overhead = r["results"].get("overhead_pct")
        overhead_str = f"{overhead:+.1f}%" if overhead is not None else "baseline"
        print(
            f"{r['hook_config']:<12} {r['batch_size']:>6} "
            f"{s['mean_ms']:>10.1f} {s['p90_ms']:>10.1f} {overhead_str:>10}"
        )

    # Verdict
    print()
    for bs in batch_sizes:
        one_hook = next(
            (r for r in all_results if r["hook_config"] == "1_hook" and r["batch_size"] == bs),
            None,
        )
        three_hooks = next(
            (r for r in all_results if r["hook_config"] == "3_hooks" and r["batch_size"] == bs),
            None,
        )
        if (
            one_hook
            and three_hooks
            and "error" not in one_hook["stats"]
            and "error" not in three_hooks["stats"]
        ):
            diff = abs(three_hooks["stats"]["mean_ms"] - one_hook["stats"]["mean_ms"])
            pct = diff / one_hook["stats"]["mean_ms"] * 100
            verdict = "negligible" if pct < 5.0 else "significant"
            print(
                f"  batch={bs}: 3-hook vs 1-hook difference is {verdict} "
                f"({diff:.1f}ms, {pct:.1f}%)"
            )

    print(f"\n{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
