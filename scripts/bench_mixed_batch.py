#!/usr/bin/env python3
"""Benchmark: do non-steered requests in a mixed batch pay the steering cost?

In continuous batching, the forward pass is shared across all requests
in a batch. If any request has active steering vectors, the steering
op runs for the whole step. This benchmark measures whether non-steered
requests in a mixed batch pay the transitive cost.

Modes tested at a fixed batch_size:
  none_active     : 0/N requests have steering (= enabled_idle baseline)
  one_active      : 1/N requests has steering
  quarter_active  : N/4 requests have steering
  half_active     : N/2 requests have steering
  all_active      : N/N requests have steering (= per_request_1 baseline)

Key question: does one_active already pay the full cost (yes = transitive),
or does it scale with the number of active requests (no = per-request)?
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
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def run_mixed(
    model: str,
    batch_size: int,
    num_active: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
) -> dict:
    """Run a mixed batch: num_active of batch_size requests are steered."""
    from vllm import LLM, SamplingParams

    print(
        f"    Loading model (batch={batch_size}, active={num_active}/{batch_size})...",
        flush=True,
    )
    llm = LLM(
        model=model,
        enable_steering=True,
        max_steering_configs=4,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompts = make_prompts(batch_size, prompt_len)

    # Non-steered sampling params (no steering_vectors field)
    sp_unsteered = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Steered sampling params
    if num_active > 0:
        vectors = random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        sp_steered = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            steering_vectors=vectors,
        )
    else:
        sp_steered = None

    # First num_active items get steered SP, rest get unsteered SP
    sp_list = []
    for i in range(batch_size):
        if i < num_active and sp_steered is not None:
            sp_list.append(sp_steered)
        else:
            sp_list.append(sp_unsteered)

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
    parser = argparse.ArgumentParser(description="Mixed-batch steering overhead")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    bs = args.batch_size
    active_counts = [0, 1, max(1, bs // 4), max(1, bs // 2), bs]
    # Deduplicate (e.g. batch_size=4 collapses)
    active_counts = sorted(set(active_counts))
    labels = {0: "none_active", bs: "all_active"}

    print(f"Mixed-batch benchmark: {args.model}")
    print(f"Batch size: {bs}")
    print(f"Active counts tested: {active_counts}")
    print()

    all_results = []
    baseline_none = None

    for num_active in active_counts:
        label = labels.get(num_active, f"{num_active}_active")
        print(f"--- {label} ({num_active}/{bs} active) ---")

        try:
            stats = run_mixed(
                model=args.model,
                batch_size=bs,
                num_active=num_active,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
            )
            mean = stats["mean_ms"]
            p90 = stats["p90_ms"]
            if num_active == 0:
                baseline_none = mean
            delta_pct = None
            if baseline_none and baseline_none > 0:
                delta_pct = (mean - baseline_none) / baseline_none * 100
            delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "baseline"
            print(f"    mean={mean:.1f}ms p90={p90:.1f}ms overhead={delta_str}")

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            stats = {"error": "OOM"}
            delta_pct = None

        params = {
            "model": args.model,
            "batch_size": bs,
            "num_active": num_active,
            "active_fraction": num_active / bs if bs > 0 else 0,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
        }
        results_out = {
            "latency_ms": {k: v for k, v in stats.items() if k != "samples_ms"},
        }
        if delta_pct is not None:
            results_out["overhead_vs_none_active_pct"] = delta_pct

        write_result(
            benchmark="vllm.mixed_batch",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=stats.get("samples_ms"),
        )
        all_results.append({
            "label": label,
            "num_active": num_active,
            "stats": stats,
            "overhead_pct": delta_pct,
        })

    print(f"\n{'=' * 80}")
    print(f"  Mixed-Batch Summary (batch={bs}, Gemma-3-4B)")
    print(f"{'=' * 80}")
    print(f"{'mode':<16} {'active':>10} {'mean_ms':>10} {'p90_ms':>10} {'overhead':>12}")
    print(f"{'-' * 80}")
    for r in all_results:
        s = r["stats"]
        if "error" in s:
            print(f"{r['label']:<16} {r['num_active']:>10} {'OOM':>10}")
            continue
        overhead = r["overhead_pct"]
        overhead_str = f"{overhead:+.1f}%" if overhead is not None else "baseline"
        print(
            f"{r['label']:<16} {r['num_active']:>10} "
            f"{s['mean_ms']:>10.1f} {s['p90_ms']:>10.1f} {overhead_str:>12}"
        )
    print(f"{'=' * 80}")
    print(
        "\nInterpretation:\n"
        "  - If 'one_active' is close to 'all_active', the cost is TRANSITIVE:\n"
        "    any active request in a batch forces the full per-request cost on everyone.\n"
        "  - If 'one_active' scales with active count toward 'all_active',\n"
        "    the cost is PROPORTIONAL to the number of steered requests.\n"
    )


if __name__ == "__main__":
    main()
