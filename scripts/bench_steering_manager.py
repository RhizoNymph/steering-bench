#!/usr/bin/env python3
"""Microbenchmark: SteeringManager Python-side overhead.

Times register_config, release_config, populate_steering_tables,
and get_row_for_config across varying layer counts and config counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn

from steering_bench.output import write_result
from steering_bench.timing import cpu_timer
from steering_bench.vectors import random_steering_vectors

try:
    from vllm.v1.worker.steering_manager import SteeringManager
    from vllm.model_executor.layers.steering import register_steering_buffers
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


def create_mock_layers(
    num_layers: int,
    hidden_size: int,
    max_steering_tokens: int,
    max_steering_configs: int,
    device: str,
) -> dict[int, nn.Module]:
    """Create mock decoder layers with steering buffers."""
    layers = {}
    for i in range(num_layers):
        mod = nn.Module()
        register_steering_buffers(
            mod,
            hidden_size,
            max_steering_tokens=max_steering_tokens,
            max_steering_configs=max_steering_configs,
        )
        # Move buffers to device
        for name, buf in list(mod.named_buffers()):
            mod.register_buffer(name, buf.to(device), persistent=False)
        layers[i] = mod
    return layers


def bench_register_release(
    manager: SteeringManager,
    vectors_list: list[dict[str, dict[int, list[float]]]],
    warmup: int,
    iters: int,
) -> tuple[dict, dict]:
    """Benchmark register_config and release_config cycle."""
    hash_base = 1000

    def register_all():
        for i, vecs in enumerate(vectors_list):
            manager.register_config(hash_base + i, vecs, phase="prefill")

    def release_all():
        for i in range(len(vectors_list)):
            manager.release_config(hash_base + i, phase="prefill")

    def register_release_cycle():
        for i, vecs in enumerate(vectors_list):
            manager.register_config(hash_base + i, vecs, phase="prefill")
        for i in range(len(vectors_list)):
            manager.release_config(hash_base + i, phase="prefill")

    reg_stats = cpu_timer(warmup, iters, register_release_cycle)
    return reg_stats.to_dict(), reg_stats.to_dict()


def bench_populate(
    manager: SteeringManager,
    steerable_layers: dict[int, nn.Module],
    warmup: int,
    iters: int,
) -> dict:
    """Benchmark populate_steering_tables."""
    stats = cpu_timer(warmup, iters, manager.populate_steering_tables, steerable_layers)
    return stats.to_dict()


def bench_get_row(
    manager: SteeringManager,
    config_hashes: list[int],
    warmup: int,
    iters: int,
) -> dict:
    """Benchmark get_row_for_config lookups."""
    def lookup_all():
        for h in config_hashes:
            manager.get_row_for_config(h, is_prefill=True)
            manager.get_row_for_config(h, is_prefill=False)

    stats = cpu_timer(warmup, iters, lookup_all)
    return stats.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Benchmark SteeringManager overhead")
    parser.add_argument("--output-dir", default="results/micro/")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not HAS_VLLM:
        print("ERROR: vLLM not installed. This benchmark requires vLLM internals.")
        sys.exit(1)

    hidden_size = 2560  # Gemma-3-4B
    max_steering_tokens = 8192

    layer_counts = [26, 34, 42]
    config_counts = [1, 4, 8, 16]
    hook_point_sets = [
        (["post_mlp"], "1_hook"),
        (["post_attn", "post_mlp"], "2_hooks"),
        (["pre_attn", "post_attn", "post_mlp"], "3_hooks"),
    ]

    total = len(layer_counts) * len(config_counts) * len(hook_point_sets)
    print(f"SteeringManager benchmark")
    print(f"Device: {args.device}")
    print(f"Sweep: {total} configurations")
    print()

    all_results = []
    idx = 0

    for num_layers in layer_counts:
        for num_configs in config_counts:
            for hook_points, hp_label in hook_point_sets:
                idx += 1
                label = f"[{idx}/{total}] layers={num_layers} configs={num_configs} {hp_label}"
                print(f"  {label} ...", end=" ", flush=True)

                # Create manager and mock layers
                manager = SteeringManager(
                    max_steering_configs=max(num_configs, 4),
                    device=torch.device(args.device),
                )
                layers = create_mock_layers(
                    num_layers, hidden_size, max_steering_tokens,
                    max(num_configs, 4), args.device,
                )

                # Generate vectors
                vectors_list = []
                for i in range(num_configs):
                    vecs = random_steering_vectors(
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        hook_points=hook_points,
                        scale=0.1,
                        seed=42 + i,
                    )
                    vectors_list.append(vecs)

                # Register configs for populate/get_row benchmarks
                config_hashes = []
                for i, vecs in enumerate(vectors_list):
                    h = 100 + i
                    manager.register_config(h, vecs, phase="prefill")
                    config_hashes.append(h)

                # Benchmark populate
                populate_stats = bench_populate(manager, layers, args.warmup, args.iters)

                # Benchmark get_row
                get_row_stats = bench_get_row(manager, config_hashes, args.warmup, args.iters)

                # Cleanup configs for register/release benchmark
                for h in config_hashes:
                    manager.release_config(h, phase="prefill")

                # Benchmark register/release cycle
                cycle_stats, _ = bench_register_release(
                    manager, vectors_list, args.warmup, args.iters
                )

                print(
                    f"populate={populate_stats['mean_ms']:.4f}ms "
                    f"get_row={get_row_stats['mean_ms']:.4f}ms "
                    f"reg_cycle={cycle_stats['mean_ms']:.4f}ms"
                )

                params = {
                    "num_layers": num_layers,
                    "num_configs": num_configs,
                    "hook_points": hp_label,
                    "hidden_size": hidden_size,
                    "max_steering_tokens": max_steering_tokens,
                }
                results = {
                    "populate_ms": {
                        k: v for k, v in populate_stats.items() if k != "samples_ms"
                    },
                    "get_row_ms": {
                        k: v for k, v in get_row_stats.items() if k != "samples_ms"
                    },
                    "register_release_cycle_ms": {
                        k: v for k, v in cycle_stats.items() if k != "samples_ms"
                    },
                }

                write_result(
                    benchmark="micro.steering_manager",
                    parameters=params,
                    results=results,
                    output_dir=args.output_dir,
                    tag=args.tag,
                )
                all_results.append({"params": params, "results": results})

                # Cleanup
                del manager, layers
                torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 90}")
    print("  SteeringManager Benchmark Summary")
    print(f"{'=' * 90}")
    print(
        f"{'layers':>8} {'configs':>8} {'hooks':>8} "
        f"{'populate':>12} {'get_row':>12} {'reg_cycle':>12}"
    )
    print(f"{'-' * 90}")
    for r in all_results:
        p = r["params"]
        print(
            f"{p['num_layers']:>8} {p['num_configs']:>8} {p['hook_points']:>8} "
            f"{r['results']['populate_ms']['mean_ms']:>10.4f}ms "
            f"{r['results']['get_row_ms']['mean_ms']:>10.4f}ms "
            f"{r['results']['register_release_cycle_ms']['mean_ms']:>10.4f}ms"
        )
    print(f"{'=' * 90}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
