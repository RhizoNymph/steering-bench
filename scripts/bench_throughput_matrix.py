#!/usr/bin/env python3
"""Throughput matrix benchmark: mode x batch_size.

Tests throughput across:
  - disabled (enable_steering=False)
  - enabled_idle (enable_steering=True, 0 active requests)
  - mixed_25, mixed_50, mixed_75 (steering enabled, fraction of batch steered)
  - all_steered (enable_steering=True, 100% active)

At each configured batch size. Loads each model (disabled / steering-enabled)
exactly once and sweeps batch sizes inside to minimize cold-start overhead.
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
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def measure_one(
    llm,
    prompts: list[str],
    sp_list,
    warmup: int,
    iters: int,
    prompt_len: int,
) -> dict:
    """Run warmup + measured iters, return latency + throughput stats."""
    # Warmup
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    samples_ms = []
    output_tokens_per_iter = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
        output_tokens_per_iter.append(sum(len(o.outputs[0].token_ids) for o in outputs))

    latency_stats = compute_stats(samples_ms).to_dict()
    avg_output_tokens = sum(output_tokens_per_iter) / len(output_tokens_per_iter)
    total_tokens_per_iter = len(prompts) * prompt_len + avg_output_tokens
    throughput_samples = [total_tokens_per_iter / (ms / 1000.0) for ms in samples_ms]
    throughput_stats = {
        k.replace("_ms", "_tps"): v
        for k, v in compute_stats(throughput_samples).to_dict().items()
        if k != "samples_ms"
    }

    return {
        "latency": latency_stats,
        "throughput": throughput_stats,
        "avg_output_tokens": avg_output_tokens,
        "total_tokens_per_iter": total_tokens_per_iter,
    }


def build_sp_list_mixed(
    batch_size: int,
    num_active: int,
    max_tokens: int,
    steering_vectors,
):
    """Build a SamplingParams list where the first num_active entries are steered."""
    from vllm import SamplingParams

    sp_unsteered = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    sp_steered = (
        SamplingParams(
            max_tokens=max_tokens, temperature=0.0, steering_vectors=steering_vectors
        )
        if num_active > 0
        else None
    )
    sp_list = []
    for i in range(batch_size):
        if i < num_active and sp_steered is not None:
            sp_list.append(sp_steered)
        else:
            sp_list.append(sp_unsteered)
    return sp_list


def main():
    parser = argparse.ArgumentParser(description="Throughput matrix: mode x batch_size")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,32",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--fractions",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated fractions of batch that are actively steered",
    )
    parser.add_argument("--max-steering-configs", type=int, default=4)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    model_config = MODEL_CONFIGS.get(
        args.model, {"hidden_size": 2560, "num_layers": 34}
    )

    print(f"Throughput matrix benchmark: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Steered fractions: {fractions}")
    print(f"prompt_len={args.prompt_len}, max_tokens={args.max_tokens}")
    print()

    # Pre-generate one steering vector (shared across all steered requests)
    steering_vectors = random_steering_vectors(
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        hook_points=["post_mlp"],
        scale=0.1,
        seed=42,
    )

    # results_map[(mode, batch_size)] = {latency, throughput, ...}
    results_map: dict = {}

    # ── Phase 1: disabled mode ──────────────────────────────────────────────
    from vllm import LLM

    print("=" * 70)
    print("  Phase 1: enable_steering=False")
    print("=" * 70)
    print("  Loading model (disabled)...", flush=True)
    llm = LLM(
        model=args.model,
        enable_steering=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    for bs in batch_sizes:
        print(f"\n  batch_size={bs}")
        prompts = make_prompts(bs, args.prompt_len)
        sp_list = build_sp_list_mixed(bs, 0, args.max_tokens, None)
        try:
            result = measure_one(llm, prompts, sp_list, args.warmup, args.iters, args.prompt_len)
            lat = result["latency"]["mean_ms"]
            tps = result["throughput"]["mean_tps"]
            print(f"    disabled: latency={lat:.1f}ms throughput={tps:.0f} tok/s")
            results_map[("disabled", bs)] = result

            params = {
                "model": args.model,
                "mode": "disabled",
                "batch_size": bs,
                "num_active": 0,
                "active_fraction": 0.0,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
                "enable_steering": False,
            }
            write_result(
                benchmark="vllm.throughput_matrix",
                parameters=params,
                results={
                    "latency_ms": {
                        k: v for k, v in result["latency"].items() if k != "samples_ms"
                    },
                    "throughput_tokens_per_sec": result["throughput"],
                    "avg_output_tokens": result["avg_output_tokens"],
                },
                output_dir=args.output_dir,
                tag=args.tag,
                raw_samples_ms=result["latency"].get("samples_ms"),
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            results_map[("disabled", bs)] = {"error": "OOM"}

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: enable_steering=True ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  Phase 2: enable_steering=True (sweep fractions)")
    print("=" * 70)
    print("  Loading model (steering enabled)...", flush=True)
    llm = LLM(
        model=args.model,
        enable_steering=True,
        max_steering_configs=args.max_steering_configs,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    for bs in batch_sizes:
        print(f"\n  batch_size={bs}")
        prompts = make_prompts(bs, args.prompt_len)

        # Deduplicate num_active values within this batch size
        seen_active = set()
        for frac in fractions:
            num_active = round(frac * bs)
            if num_active in seen_active:
                continue
            seen_active.add(num_active)

            if num_active == 0:
                mode = "enabled_idle"
            elif num_active == bs:
                mode = "all_steered"
            else:
                mode = f"mixed_{round(frac * 100)}"

            sp_list = build_sp_list_mixed(bs, num_active, args.max_tokens, steering_vectors)

            try:
                result = measure_one(
                    llm, prompts, sp_list, args.warmup, args.iters, args.prompt_len
                )
                lat = result["latency"]["mean_ms"]
                tps = result["throughput"]["mean_tps"]
                print(
                    f"    {mode:<14} ({num_active}/{bs}): "
                    f"latency={lat:.1f}ms throughput={tps:.0f} tok/s"
                )
                results_map[(mode, bs)] = result

                params = {
                    "model": args.model,
                    "mode": mode,
                    "batch_size": bs,
                    "num_active": num_active,
                    "active_fraction": num_active / bs,
                    "prompt_len": args.prompt_len,
                    "max_tokens": args.max_tokens,
                    "enable_steering": True,
                    "max_steering_configs": args.max_steering_configs,
                }
                write_result(
                    benchmark="vllm.throughput_matrix",
                    parameters=params,
                    results={
                        "latency_ms": {
                            k: v for k, v in result["latency"].items() if k != "samples_ms"
                        },
                        "throughput_tokens_per_sec": result["throughput"],
                        "avg_output_tokens": result["avg_output_tokens"],
                    },
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=result["latency"].get("samples_ms"),
                )
            except torch.cuda.OutOfMemoryError:
                print(f"    {mode:<14} ({num_active}/{bs}): OOM!")
                results_map[(mode, bs)] = {"error": "OOM"}

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary tables ──────────────────────────────────────────────────────
    mode_order = ["disabled", "enabled_idle", "mixed_25", "mixed_50", "mixed_75", "all_steered"]

    def get(mode, bs, field):
        r = results_map.get((mode, bs))
        if r is None or "error" in r:
            return None
        if field == "latency":
            return r["latency"]["mean_ms"]
        if field == "throughput":
            return r["throughput"]["mean_tps"]
        return None

    # Latency table
    print(f"\n{'=' * 90}")
    print(f"  Latency (ms) by mode x batch_size")
    print(f"{'=' * 90}")
    header = f"{'mode':<16}"
    for bs in batch_sizes:
        header += f" {f'b={bs}':>12}"
    print(header)
    print(f"{'-' * 90}")
    for mode in mode_order:
        if not any((mode, bs) in results_map for bs in batch_sizes):
            continue
        row = f"{mode:<16}"
        for bs in batch_sizes:
            val = get(mode, bs, "latency")
            row += f" {f'{val:.1f}' if val is not None else '—':>12}"
        print(row)

    # Throughput table
    print(f"\n{'=' * 90}")
    print(f"  Throughput (tok/s) by mode x batch_size")
    print(f"{'=' * 90}")
    header = f"{'mode':<16}"
    for bs in batch_sizes:
        header += f" {f'b={bs}':>12}"
    print(header)
    print(f"{'-' * 90}")
    for mode in mode_order:
        if not any((mode, bs) in results_map for bs in batch_sizes):
            continue
        row = f"{mode:<16}"
        for bs in batch_sizes:
            val = get(mode, bs, "throughput")
            row += f" {f'{val:.0f}' if val is not None else '—':>12}"
        print(row)

    # Overhead vs disabled
    print(f"\n{'=' * 90}")
    print(f"  Latency overhead (%) vs disabled at same batch size")
    print(f"{'=' * 90}")
    header = f"{'mode':<16}"
    for bs in batch_sizes:
        header += f" {f'b={bs}':>12}"
    print(header)
    print(f"{'-' * 90}")
    for mode in mode_order:
        if mode == "disabled":
            continue
        if not any((mode, bs) in results_map for bs in batch_sizes):
            continue
        row = f"{mode:<16}"
        for bs in batch_sizes:
            mode_val = get(mode, bs, "latency")
            baseline = get("disabled", bs, "latency")
            if mode_val is not None and baseline is not None and baseline > 0:
                overhead = (mode_val - baseline) / baseline * 100
                row += f" {f'{overhead:+.1f}%':>12}"
            else:
                row += f" {'—':>12}"
        print(row)

    # Throughput loss vs disabled
    print(f"\n{'=' * 90}")
    print(f"  Throughput loss (%) vs disabled at same batch size")
    print(f"{'=' * 90}")
    header = f"{'mode':<16}"
    for bs in batch_sizes:
        header += f" {f'b={bs}':>12}"
    print(header)
    print(f"{'-' * 90}")
    for mode in mode_order:
        if mode == "disabled":
            continue
        if not any((mode, bs) in results_map for bs in batch_sizes):
            continue
        row = f"{mode:<16}"
        for bs in batch_sizes:
            mode_val = get(mode, bs, "throughput")
            baseline = get("disabled", bs, "throughput")
            if mode_val is not None and baseline is not None and baseline > 0:
                loss = (baseline - mode_val) / baseline * 100
                row += f" {f'{loss:+.1f}%':>12}"
            else:
                row += f" {'—':>12}"
        print(row)

    print(f"\n{'=' * 90}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
