#!/usr/bin/env python3
"""Table sizing matrix: max_steering_configs × batch_size × distinct_configs.

Measures whether giving the steering table more headroom (larger
max_steering_configs) reduces per-request overhead, and how that
interacts with both batch size and the number of distinct steering
configs in the workload.

Default sweep produces 2 × 3 × 3 = 18 steered cells + 3 disabled
baselines. Loads model once per (enable_steering, max_steering_configs)
combo to minimize cold-start overhead.
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

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
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


def main():
    parser = argparse.ArgumentParser(
        description="Table sizing matrix: max_cfg × batch × distinct"
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--max-configs-sweep",
        default="4,16",
        help="Comma-separated max_steering_configs values to sweep",
    )
    parser.add_argument(
        "--distinct-sweep",
        default="1,4,8",
        help="Comma-separated distinct_configs values (unique vectors per batch)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="8,16,32",
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--tag", default="table-sizing")
    args = parser.parse_args()

    max_configs_sweep = [int(x) for x in args.max_configs_sweep.split(",")]
    distinct_sweep = [int(x) for x in args.distinct_sweep.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    model_config = MODEL_CONFIGS.get(
        args.model, {"hidden_size": 2560, "num_layers": 34}
    )

    print(f"Table sizing benchmark: {args.model}")
    print(f"max_steering_configs: {max_configs_sweep}")
    print(f"distinct_configs:     {distinct_sweep}")
    print(f"batch_sizes:          {batch_sizes}")
    print(f"prompt_len={args.prompt_len}, max_tokens={args.max_tokens}")
    print()

    # Collect results keyed by (max_cfg_or_None, batch_size, distinct_or_None)
    # None values indicate the disabled baseline.
    results_map: dict = {}

    # ── Phase 1: disabled baselines ─────────────────────────────────────────
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("  Phase 1: disabled baseline")
    print("=" * 70)
    print("  Loading model (enable_steering=False)...", flush=True)
    llm = LLM(
        model=args.model,
        enable_steering=False,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    sp_unsteered = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    for bs in batch_sizes:
        prompts = make_prompts(bs, args.prompt_len)
        sp_list = [sp_unsteered] * bs
        print(f"\n  batch_size={bs}", flush=True)
        try:
            result = measure_one(
                llm, prompts, sp_list, args.warmup, args.iters, args.prompt_len
            )
            lat = result["latency"]["mean_ms"]
            tps = result["throughput"]["mean_tps"]
            print(f"    disabled: latency={lat:.1f}ms throughput={tps:.0f} tok/s")
            results_map[(0, bs, 0)] = result

            params = {
                "model": args.model,
                "mode": "disabled",
                "max_steering_configs": 0,
                "batch_size": bs,
                "distinct_configs": 0,
                "prompt_len": args.prompt_len,
                "max_tokens": args.max_tokens,
                "enable_steering": False,
            }
            write_result(
                benchmark="vllm.table_sizing",
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
            results_map[(0, bs, 0)] = {"error": "OOM"}

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: steering, one model load per max_steering_configs ──────────
    for max_cfg in max_configs_sweep:
        print()
        print("=" * 70)
        print(f"  Phase 2: enable_steering=True, max_steering_configs={max_cfg}")
        print("=" * 70)
        print("  Loading model...", flush=True)
        llm = LLM(
            model=args.model,
            enable_steering=True,
            max_steering_configs=max_cfg,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )

        for bs in batch_sizes:
            prompts = make_prompts(bs, args.prompt_len)
            print(f"\n  batch_size={bs}")
            for distinct in distinct_sweep:
                if distinct > bs:
                    continue
                if distinct > max_cfg:
                    # Workload wants more distinct configs than the table can
                    # hold. Skip to avoid register_config RuntimeError.
                    print(f"    distinct={distinct}: SKIPPED (> max_cfg={max_cfg})")
                    continue

                diverse = random_steering_vectors_diverse(
                    hidden_size=model_config["hidden_size"],
                    num_layers=model_config["num_layers"],
                    num_configs=distinct,
                    hook_points=["post_mlp"],
                    scale=0.1,
                    base_seed=42,
                )
                sp_list = [
                    SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        steering_vectors=diverse[i % distinct],
                    )
                    for i in range(bs)
                ]

                try:
                    result = measure_one(
                        llm, prompts, sp_list, args.warmup, args.iters, args.prompt_len
                    )
                    lat = result["latency"]["mean_ms"]
                    tps = result["throughput"]["mean_tps"]
                    print(
                        f"    distinct={distinct}: "
                        f"latency={lat:.1f}ms throughput={tps:.0f} tok/s"
                    )
                    results_map[(max_cfg, bs, distinct)] = result

                    params = {
                        "model": args.model,
                        "mode": "steered",
                        "max_steering_configs": max_cfg,
                        "batch_size": bs,
                        "distinct_configs": distinct,
                        "prompt_len": args.prompt_len,
                        "max_tokens": args.max_tokens,
                        "enable_steering": True,
                    }
                    write_result(
                        benchmark="vllm.table_sizing",
                        parameters=params,
                        results={
                            "latency_ms": {
                                k: v
                                for k, v in result["latency"].items()
                                if k != "samples_ms"
                            },
                            "throughput_tokens_per_sec": result["throughput"],
                            "avg_output_tokens": result["avg_output_tokens"],
                        },
                        output_dir=args.output_dir,
                        tag=args.tag,
                        raw_samples_ms=result["latency"].get("samples_ms"),
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"    distinct={distinct}: OOM!")
                    results_map[(max_cfg, bs, distinct)] = {"error": "OOM"}

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary tables ──────────────────────────────────────────────────────
    def cell(max_cfg, bs, distinct, field):
        r = results_map.get((max_cfg, bs, distinct))
        if r is None or "error" in r:
            return None
        if field == "latency":
            return r["latency"]["mean_ms"]
        if field == "throughput":
            return r["throughput"]["mean_tps"]
        return None

    # Throughput loss % vs disabled, by distinct x (max_cfg, batch_size)
    print(f"\n{'=' * 100}")
    print(f"  Throughput loss (%) vs disabled")
    print(f"{'=' * 100}")

    header = f"{'distinct':<10}"
    for max_cfg in max_configs_sweep:
        for bs in batch_sizes:
            header += f" {f'max={max_cfg} b={bs}':>14}"
    print(header)
    print(f"{'-' * 100}")

    for distinct in distinct_sweep:
        row = f"{distinct:<10}"
        for max_cfg in max_configs_sweep:
            for bs in batch_sizes:
                if distinct > bs or distinct > max_cfg:
                    row += f" {'—':>14}"
                    continue
                base_tps = cell(0, bs, 0, "throughput")
                tps = cell(max_cfg, bs, distinct, "throughput")
                if base_tps and tps and base_tps > 0:
                    loss = (base_tps - tps) / base_tps * 100
                    row += f" {f'{loss:+.1f}%':>14}"
                else:
                    row += f" {'—':>14}"
        print(row)

    # Interaction delta: loss(max=4) - loss(max=16) = "benefit of bigger table"
    if len(max_configs_sweep) >= 2:
        small_max = min(max_configs_sweep)
        large_max = max(max_configs_sweep)
        print(f"\n{'=' * 100}")
        print(
            f"  Throughput benefit of max_cfg={large_max} vs {small_max} "
            f"(percentage points of loss reduction)"
        )
        print(f"{'=' * 100}")
        header = f"{'distinct':<10}"
        for bs in batch_sizes:
            header += f" {f'batch={bs}':>14}"
        print(header)
        print(f"{'-' * 100}")
        for distinct in distinct_sweep:
            row = f"{distinct:<10}"
            for bs in batch_sizes:
                if distinct > bs or distinct > small_max:
                    row += f" {'—':>14}"
                    continue
                base_tps = cell(0, bs, 0, "throughput")
                small_tps = cell(small_max, bs, distinct, "throughput")
                large_tps = cell(large_max, bs, distinct, "throughput")
                if base_tps and small_tps and large_tps and base_tps > 0:
                    small_loss = (base_tps - small_tps) / base_tps * 100
                    large_loss = (base_tps - large_tps) / base_tps * 100
                    delta = small_loss - large_loss
                    row += f" {f'{delta:+.1f}pp':>14}"
                else:
                    row += f" {'—':>14}"
            print(row)

    print(f"\n{'=' * 100}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
