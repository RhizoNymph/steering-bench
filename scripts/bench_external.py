#!/usr/bin/env python3
"""Cross-library steering comparison.

Runs Tier 1 (single-request) and Tier 2 (batched N=16) benchmarks
across all installed steering libraries. Discovers available libraries
at runtime and skips those not installed.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.external.base import cleanup_gpu, is_library_available
from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import random_steering_vectors

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}

# Library name -> (import check, class import path)
LIBRARY_REGISTRY: list[tuple[str, str | None, str, str]] = [
    # (name, required_package, module_path, class_name)
    ("hf_baseline", None, "steering_bench.external.hf_baseline", "HFBaselineBenchmark"),
    ("transformerlens", "transformer_lens", "steering_bench.external.transformerlens_bench", "TransformerLensBenchmark"),
    ("nnsight", "nnsight", "steering_bench.external.nnsight_bench", "NnsightBenchmark"),
    ("repeng", "repeng", "steering_bench.external.repeng_bench", "RepengBenchmark"),
    ("pyvene", "pyvene", "steering_bench.external.pyvene_bench", "PyveneBenchmark"),
    ("vllm_single", "vllm", "steering_bench.external.vllm_single", "VllmSingleBenchmark"),
    ("vllm_batched", "vllm", "steering_bench.external.vllm_batched", "VllmBatchedBenchmark"),
]


def discover_libraries(
    filter_names: list[str] | None = None,
) -> list[tuple[str, type]]:
    """Discover which benchmark libraries are available."""
    import importlib

    available = []
    for name, required_pkg, module_path, class_name in LIBRARY_REGISTRY:
        if filter_names and name not in filter_names:
            continue
        if required_pkg and not is_library_available(required_pkg):
            print(f"  {name}: SKIPPED ({required_pkg} not installed)")
            continue
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            available.append((name, cls))
            print(f"  {name}: available")
        except (ImportError, AttributeError) as e:
            print(f"  {name}: SKIPPED ({e})")
    return available


def run_tier1(
    bench_cls: type,
    name: str,
    model_id: str,
    vector: list[float],
    layer: int,
    hook: str,
    prompt: str,
    max_tokens: int,
    warmup: int,
    iters: int,
) -> dict:
    """Run Tier 1 (single-request) benchmark for one library."""
    bench = bench_cls()
    print(f"    Setting up {name}...", flush=True)
    bench.setup(model_id, vector, layer, hook)
    memory_mb = bench.memory_allocated_mb()

    # Warmup
    print(f"    Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        bench.generate_single(prompt, max_tokens)

    # Measure
    print(f"    Measuring ({iters} iters)...", flush=True)
    samples = []
    output_tokens_list = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_tokens = bench.generate_single(prompt, max_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
        output_tokens_list.append(n_tokens)

    bench.teardown()

    stats = compute_stats(samples)
    avg_tokens = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
    tokens_per_sec = avg_tokens / (stats.mean_ms / 1000.0) if stats.mean_ms > 0 else 0

    return {
        "latency": stats.to_dict(),
        "memory_mb": memory_mb,
        "avg_output_tokens": avg_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def run_tier2(
    bench_cls: type,
    name: str,
    model_id: str,
    vectors: list[list[float]],
    layer: int,
    hook: str,
    prompts: list[str],
    max_tokens: int,
    warmup: int,
    iters: int,
) -> dict:
    """Run Tier 2 (batched) benchmark for one library."""
    # Use the first vector for setup
    bench = bench_cls()
    print(f"    Setting up {name} (batched)...", flush=True)
    bench.setup(model_id, vectors[0], layer, hook)
    memory_mb = bench.memory_allocated_mb()

    # Warmup
    print(f"    Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        bench.generate_batch(prompts, vectors, max_tokens)

    # Measure
    print(f"    Measuring ({iters} iters, batch={len(prompts)})...", flush=True)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_tokens = bench.generate_batch(prompts, vectors, max_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    bench.teardown()

    stats = compute_stats(samples)
    req_per_sec = len(prompts) / (stats.mean_ms / 1000.0) if stats.mean_ms > 0 else 0

    return {
        "batch_latency": stats.to_dict(),
        "memory_mb": memory_mb,
        "batch_size": len(prompts),
        "req_per_sec": req_per_sec,
        "avg_per_request_ms": stats.mean_ms / len(prompts),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-library steering comparison")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="results/external/")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--hook", default="post_mlp")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--libraries", default="all", help="Comma-separated library names or 'all'")
    parser.add_argument("--skip-tier1", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2048, "num_layers": 16})
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_layers"]

    # Generate steering vector for the target layer
    all_vectors = random_steering_vectors(
        hidden_size=hidden_size,
        num_layers=num_layers,
        hook_points=[args.hook],
        scale=0.1,
        seed=42,
    )
    # Extract flat vector for the target layer
    single_vector = all_vectors[args.hook][args.layer]

    # Generate diverse vectors for Tier 2
    diverse_vectors = [
        random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=[args.hook],
            scale=0.1,
            seed=42 + i,
        )[args.hook][args.layer]
        for i in range(args.batch_size)
    ]

    # Prompt
    words_needed = max(1, int(args.prompt_len / 1.3))
    prompt = " ".join(["hello"] * words_needed)
    prompts = [prompt] * args.batch_size

    # Discover libraries
    filter_names = None if args.libraries == "all" else args.libraries.split(",")
    print("Discovering available libraries:")
    available = discover_libraries(filter_names)
    print(f"\n{len(available)} libraries available\n")

    if not available:
        print("No libraries available. Install optional dependencies.")
        sys.exit(1)

    # --- Tier 1: Single-request ---
    tier1_results = []
    if not args.skip_tier1:
        print(f"{'=' * 70}")
        print(f"  Tier 1: Single-Request Comparison ({args.model})")
        print(f"{'=' * 70}")

        for name, cls in available:
            print(f"\n--- {name} ---")
            try:
                result = run_tier1(
                    cls, name, args.model, single_vector, args.layer, args.hook,
                    prompt, args.max_tokens, args.warmup, args.iters,
                )
                lat = result["latency"]
                print(
                    f"    mean={lat['mean_ms']:.1f}ms "
                    f"tokens/sec={result['tokens_per_sec']:.0f} "
                    f"memory={result['memory_mb']:.0f}MB"
                )

                params = {
                    "model": args.model,
                    "library": name,
                    "layer": args.layer,
                    "hook": args.hook,
                    "max_tokens": args.max_tokens,
                    "prompt_len": args.prompt_len,
                }
                results_out = {
                    "latency_ms": {k: v for k, v in lat.items() if k != "samples_ms"},
                    "memory_mb": result["memory_mb"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "avg_output_tokens": result["avg_output_tokens"],
                }
                write_result(
                    benchmark=f"external.tier1.{name}",
                    parameters=params,
                    results=results_out,
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=lat.get("samples_ms"),
                )
                tier1_results.append({"name": name, **results_out})

            except Exception as e:
                print(f"    FAILED: {e}")
                tier1_results.append({"name": name, "error": str(e)})

            cleanup_gpu()

    # --- Tier 2: Batched ---
    tier2_results = []
    if not args.skip_tier2:
        print(f"\n{'=' * 70}")
        print(f"  Tier 2: Batched Comparison (N={args.batch_size}, {args.model})")
        print(f"{'=' * 70}")

        for name, cls in available:
            print(f"\n--- {name} ---")
            try:
                result = run_tier2(
                    cls, name, args.model, diverse_vectors, args.layer, args.hook,
                    prompts, args.max_tokens, args.warmup, args.iters,
                )
                lat = result["batch_latency"]
                print(
                    f"    batch_mean={lat['mean_ms']:.0f}ms "
                    f"req/sec={result['req_per_sec']:.1f} "
                    f"per_req={result['avg_per_request_ms']:.1f}ms"
                )

                params = {
                    "model": args.model,
                    "library": name,
                    "layer": args.layer,
                    "hook": args.hook,
                    "max_tokens": args.max_tokens,
                    "batch_size": args.batch_size,
                }
                results_out = {
                    "batch_latency_ms": {k: v for k, v in lat.items() if k != "samples_ms"},
                    "memory_mb": result["memory_mb"],
                    "req_per_sec": result["req_per_sec"],
                    "avg_per_request_ms": result["avg_per_request_ms"],
                }
                write_result(
                    benchmark=f"external.tier2.{name}",
                    parameters=params,
                    results=results_out,
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=lat.get("samples_ms"),
                )
                tier2_results.append({"name": name, **results_out})

            except Exception as e:
                print(f"    FAILED: {e}")
                tier2_results.append({"name": name, "error": str(e)})

            cleanup_gpu()

    # --- Summary ---
    if tier1_results:
        print(f"\n{'=' * 80}")
        print(f"  Tier 1 Summary: Single-Request")
        print(f"{'=' * 80}")
        print(f"{'library':<20} {'mean_ms':>10} {'p90_ms':>10} {'tok/sec':>10} {'mem_MB':>10}")
        print(f"{'-' * 80}")
        # Sort by mean latency
        sorted_t1 = sorted(
            tier1_results, key=lambda r: r.get("latency_ms", {}).get("mean_ms", float("inf"))
        )
        baseline_mean = next(
            (r["latency_ms"]["mean_ms"] for r in sorted_t1 if r["name"] == "hf_baseline" and "error" not in r),
            None,
        )
        for r in sorted_t1:
            if "error" in r:
                print(f"{r['name']:<20} {'FAILED':>10}")
                continue
            lat = r["latency_ms"]
            speedup = ""
            if baseline_mean and baseline_mean > 0:
                speedup = f" ({lat['mean_ms']/baseline_mean:.2f}x baseline)"
            print(
                f"{r['name']:<20} {lat['mean_ms']:>10.1f} {lat['p90_ms']:>10.1f} "
                f"{r['tokens_per_sec']:>10.0f} {r['memory_mb']:>10.0f}{speedup}"
            )

    if tier2_results:
        print(f"\n{'=' * 80}")
        print(f"  Tier 2 Summary: Batched (N={args.batch_size})")
        print(f"{'=' * 80}")
        print(f"{'library':<20} {'batch_ms':>10} {'req/sec':>10} {'per_req':>10} {'mem_MB':>10}")
        print(f"{'-' * 80}")
        sorted_t2 = sorted(
            tier2_results, key=lambda r: r.get("batch_latency_ms", {}).get("mean_ms", float("inf"))
        )
        for r in sorted_t2:
            if "error" in r:
                print(f"{r['name']:<20} {'FAILED':>10}")
                continue
            lat = r["batch_latency_ms"]
            print(
                f"{r['name']:<20} {lat['mean_ms']:>10.0f} {r['req_per_sec']:>10.1f} "
                f"{r['avg_per_request_ms']:>10.1f} {r['memory_mb']:>10.0f}"
            )

    print(f"\n{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
