#!/usr/bin/env python3
"""Microbenchmark: steering_index construction loop.

Times the CPU loop that fills the steering_index tensor, which maps
each token position to its steering table row. This runs on CPU
touching a GPU tensor, and can become a bottleneck at high batch sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import write_result
from steering_bench.timing import cpu_timer


def build_index_uniform(
    steering_index: torch.Tensor,
    num_requests: int,
    tokens_per_request: int,
    row_assignments: list[int],
) -> None:
    """Simulate uniform index building: all requests same token count.

    This mirrors the model runner's inner loop:
    for each request, write the assigned row index for all its tokens.
    """
    offset = 0
    for i in range(num_requests):
        row = row_assignments[i % len(row_assignments)]
        steering_index[offset : offset + tokens_per_request] = row
        offset += tokens_per_request


def build_index_mixed_phase(
    steering_index: torch.Tensor,
    prefill_requests: int,
    prefill_tokens: int,
    decode_requests: int,
    prefill_rows: list[int],
    decode_rows: list[int],
) -> None:
    """Simulate mixed-phase index building.

    Prefill requests get many tokens each; decode requests get 1 token each.
    This is the realistic scenario during continuous batching.
    """
    offset = 0
    # Prefill requests
    for i in range(prefill_requests):
        row = prefill_rows[i % len(prefill_rows)]
        steering_index[offset : offset + prefill_tokens] = row
        offset += prefill_tokens
    # Decode requests (1 token each)
    for i in range(decode_requests):
        row = decode_rows[i % len(decode_rows)]
        steering_index[offset] = row
        offset += 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark steering_index construction")
    parser.add_argument("--output-dir", default="results/micro/")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    max_tokens = 8192
    steering_index = torch.zeros(max_tokens, dtype=torch.long, device=args.device)

    # Uniform sweep
    num_requests_sweep = [1, 4, 8, 16, 32, 64]
    tokens_per_request_sweep = [32, 128, 512]

    # Row assignments for different config counts
    row_configs = {
        "no_steering": [0],
        "global_only": [1],
        "1_config": [3],
        "4_configs": [3, 4, 5, 6],
    }

    total_uniform = len(num_requests_sweep) * len(tokens_per_request_sweep) * len(row_configs)

    print("Index Building Benchmark")
    print(f"Device: {args.device}")
    print(f"Max tokens buffer: {max_tokens}")
    print(f"Uniform sweep: {total_uniform} configurations")
    print()

    all_results = []
    idx = 0

    # --- Uniform benchmarks ---
    print("--- Uniform token distribution ---")
    for num_requests in num_requests_sweep:
        for tokens_per_request in tokens_per_request_sweep:
            total_tokens = num_requests * tokens_per_request
            if total_tokens > max_tokens:
                continue

            for config_name, rows in row_configs.items():
                idx += 1
                label = (
                    f"[{idx}] reqs={num_requests} toks/req={tokens_per_request} "
                    f"config={config_name}"
                )
                print(f"  {label} ...", end=" ", flush=True)

                stats = cpu_timer(
                    args.warmup,
                    args.iters,
                    build_index_uniform,
                    steering_index,
                    num_requests,
                    tokens_per_request,
                    rows,
                )
                stats_dict = stats.to_dict()
                print(f"mean={stats_dict['mean_ms']:.4f}ms")

                params = {
                    "mode": "uniform",
                    "num_requests": num_requests,
                    "tokens_per_request": tokens_per_request,
                    "total_tokens": total_tokens,
                    "config_name": config_name,
                    "num_distinct_rows": len(rows),
                }
                results = {
                    "build_index_ms": {
                        k: v for k, v in stats_dict.items() if k != "samples_ms"
                    },
                }
                write_result(
                    benchmark="micro.index_building",
                    parameters=params,
                    results=results,
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=stats_dict["samples_ms"],
                )
                all_results.append({"params": params, "results": results})

    # --- Mixed-phase benchmarks ---
    print("\n--- Mixed-phase (prefill + decode) ---")
    mixed_scenarios = [
        {"prefill_reqs": 1, "prefill_tokens": 512, "decode_reqs": 15},
        {"prefill_reqs": 2, "prefill_tokens": 256, "decode_reqs": 30},
        {"prefill_reqs": 4, "prefill_tokens": 128, "decode_reqs": 60},
        {"prefill_reqs": 0, "prefill_tokens": 0, "decode_reqs": 64},  # All decode
    ]
    prefill_rows = [3, 4, 5, 6]
    decode_rows = [3, 4, 5, 6]

    for scenario in mixed_scenarios:
        total_tokens = (
            scenario["prefill_reqs"] * scenario["prefill_tokens"]
            + scenario["decode_reqs"]
        )
        if total_tokens > max_tokens:
            continue

        label = (
            f"  prefill={scenario['prefill_reqs']}x{scenario['prefill_tokens']} "
            f"decode={scenario['decode_reqs']}x1"
        )
        print(f"{label} ...", end=" ", flush=True)

        stats = cpu_timer(
            args.warmup,
            args.iters,
            build_index_mixed_phase,
            steering_index,
            scenario["prefill_reqs"],
            scenario["prefill_tokens"],
            scenario["decode_reqs"],
            prefill_rows,
            decode_rows,
        )
        stats_dict = stats.to_dict()
        print(f"mean={stats_dict['mean_ms']:.4f}ms")

        params = {
            "mode": "mixed_phase",
            "total_tokens": total_tokens,
            **scenario,
        }
        results = {
            "build_index_ms": {
                k: v for k, v in stats_dict.items() if k != "samples_ms"
            },
        }
        write_result(
            benchmark="micro.index_building",
            parameters=params,
            results=results,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=stats_dict["samples_ms"],
        )
        all_results.append({"params": params, "results": results})

    # Summary
    print(f"\n{'=' * 80}")
    print("  Index Building Benchmark Summary")
    print(f"{'=' * 80}")
    for r in all_results:
        p = r["params"]
        lat = r["results"]["build_index_ms"]
        if p["mode"] == "uniform":
            print(
                f"  uniform reqs={p['num_requests']:>3} toks/req={p['tokens_per_request']:>4} "
                f"config={p['config_name']:<12} mean={lat['mean_ms']:.4f}ms p90={lat['p90_ms']:.4f}ms"
            )
        else:
            print(
                f"  mixed   prefill={p['prefill_reqs']}x{p['prefill_tokens']:<4} "
                f"decode={p['decode_reqs']:<3}x1  "
                f"mean={lat['mean_ms']:.4f}ms p90={lat['p90_ms']:.4f}ms"
            )
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
