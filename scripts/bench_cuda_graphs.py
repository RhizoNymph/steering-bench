#!/usr/bin/env python3
"""Ablation benchmark: CUDA graphs interaction with steering.

2x2 matrix: (enforce_eager True/False) x (enable_steering True/False)

Key question: does the opaque steering op break CUDA graph benefits?
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


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def run_config(
    model: str,
    enforce_eager: bool,
    enable_steering: bool,
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
) -> dict:
    """Run a single (enforce_eager, enable_steering, batch_size) config."""
    from vllm import LLM, SamplingParams

    label = f"eager={'Y' if enforce_eager else 'N'}_steer={'Y' if enable_steering else 'N'}"
    print(f"    Loading model ({label})...", flush=True)

    llm = LLM(
        model=model,
        enforce_eager=enforce_eager,
        enable_steering=enable_steering,
        max_steering_configs=4,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompts = make_prompts(batch_size, prompt_len)

    if enable_steering:
        vectors = random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_vectors=vectors,
        )
    else:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

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

    stats = compute_stats(samples)
    return stats.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Ablation: CUDA graphs x steering")
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

    # 2x2 matrix
    configs = [
        {"enforce_eager": False, "enable_steering": False, "label": "graphs_no_steer"},
        {"enforce_eager": False, "enable_steering": True, "label": "graphs_with_steer"},
        {"enforce_eager": True, "enable_steering": False, "label": "eager_no_steer"},
        {"enforce_eager": True, "enable_steering": True, "label": "eager_with_steer"},
    ]

    total = len(configs) * len(batch_sizes)
    print(f"CUDA Graphs Ablation: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total configurations: {total}")
    print()

    # Collect results: results_map[label][batch_size] = stats
    results_map: dict[str, dict[int, dict]] = {c["label"]: {} for c in configs}
    all_flat = []

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        for batch_size in batch_sizes:
            print(f"  batch_size={batch_size}")
            try:
                stats = run_config(
                    model=args.model,
                    enforce_eager=cfg["enforce_eager"],
                    enable_steering=cfg["enable_steering"],
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    max_tokens=args.max_tokens,
                    warmup=args.warmup,
                    iters=args.iters,
                    hidden_size=model_config["hidden_size"],
                    num_layers=model_config["num_layers"],
                )
                print(f"    mean={stats['mean_ms']:.1f}ms p90={stats['p90_ms']:.1f}ms")
                results_map[cfg["label"]][batch_size] = stats

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM!")
                results_map[cfg["label"]][batch_size] = {"error": "OOM"}

            params = {
                "model": args.model,
                "enforce_eager": cfg["enforce_eager"],
                "enable_steering": cfg["enable_steering"],
                "label": cfg["label"],
                "batch_size": batch_size,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
            }
            results_out = {
                "latency_ms": {
                    k: v for k, v in results_map[cfg["label"]][batch_size].items()
                    if k != "samples_ms"
                },
            }
            write_result(
                benchmark="ablation.cuda_graphs",
                parameters=params,
                results=results_out,
                output_dir=args.output_dir,
                tag=args.tag,
                raw_samples_ms=results_map[cfg["label"]][batch_size].get("samples_ms"),
            )
            all_flat.append({
                "label": cfg["label"],
                "batch_size": batch_size,
                "stats": results_map[cfg["label"]][batch_size],
            })

    # Derived metrics
    print(f"\n{'=' * 95}")
    print(f"  CUDA Graphs Ablation Summary: {args.model}")
    print(f"{'=' * 95}")

    # Raw latencies
    print(f"\n  Raw Latencies (mean ms):")
    print(f"  {'batch':>6} {'graphs_no_steer':>18} {'graphs_w_steer':>18} {'eager_no_steer':>18} {'eager_w_steer':>18}")
    print(f"  {'-' * 90}")
    for bs in batch_sizes:
        vals = []
        for label in ["graphs_no_steer", "graphs_with_steer", "eager_no_steer", "eager_with_steer"]:
            s = results_map[label].get(bs, {})
            if "error" in s:
                vals.append("OOM")
            else:
                vals.append(f"{s.get('mean_ms', 0):.1f}")
        print(f"  {bs:>6} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18} {vals[3]:>18}")

    # Derived analysis
    print(f"\n  Derived Metrics:")
    print(f"  {'batch':>6} {'graph_speedup':>14} {'graph_speedup':>14} {'steer_overhead':>16} {'steer_overhead':>16}")
    print(f"  {'':>6} {'(no steer)':>14} {'(w/ steer)':>14} {'(w/ graphs)':>16} {'(eager)':>16}")
    print(f"  {'-' * 90}")

    for bs in batch_sizes:
        gns = results_map["graphs_no_steer"].get(bs, {})
        gws = results_map["graphs_with_steer"].get(bs, {})
        ens = results_map["eager_no_steer"].get(bs, {})
        ews = results_map["eager_with_steer"].get(bs, {})

        def safe_mean(s):
            if isinstance(s, dict) and "mean_ms" in s:
                return s["mean_ms"]
            return None

        gns_m, gws_m, ens_m, ews_m = safe_mean(gns), safe_mean(gws), safe_mean(ens), safe_mean(ews)

        # Graph speedup without steering: eager/graphs
        gs_no = f"{ens_m / gns_m:.2f}x" if gns_m and ens_m else "N/A"
        # Graph speedup with steering: eager_steer/graphs_steer
        gs_ws = f"{ews_m / gws_m:.2f}x" if gws_m and ews_m else "N/A"
        # Steering overhead with graphs: (graphs_steer - graphs_no) / graphs_no
        so_g = f"{(gws_m - gns_m) / gns_m * 100:+.1f}%" if gns_m and gws_m else "N/A"
        # Steering overhead without graphs: (eager_steer - eager_no) / eager_no
        so_e = f"{(ews_m - ens_m) / ens_m * 100:+.1f}%" if ens_m and ews_m else "N/A"

        print(f"  {bs:>6} {gs_no:>14} {gs_ws:>14} {so_g:>16} {so_e:>16}")

    # Interaction effect
    print(f"\n  Interaction Effect:")
    for bs in batch_sizes:
        gns = results_map["graphs_no_steer"].get(bs, {})
        gws = results_map["graphs_with_steer"].get(bs, {})
        ens = results_map["eager_no_steer"].get(bs, {})
        ews = results_map["eager_with_steer"].get(bs, {})

        def safe_mean(s):
            if isinstance(s, dict) and "mean_ms" in s:
                return s["mean_ms"]
            return None

        gns_m, gws_m, ens_m, ews_m = safe_mean(gns), safe_mean(gws), safe_mean(ens), safe_mean(ews)

        if all(v is not None for v in [gns_m, gws_m, ens_m, ews_m]):
            graph_speedup_no_steer = ens_m / gns_m
            graph_speedup_w_steer = ews_m / gws_m
            interaction = graph_speedup_w_steer / graph_speedup_no_steer
            verdict = "preserved" if interaction > 0.9 else "degraded"
            print(
                f"  batch={bs}: CUDA graph benefit is {verdict} "
                f"({graph_speedup_no_steer:.2f}x without steering vs "
                f"{graph_speedup_w_steer:.2f}x with steering, "
                f"ratio={interaction:.2f})"
            )

    print(f"\n{'=' * 95}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
