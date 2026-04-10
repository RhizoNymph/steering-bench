#!/usr/bin/env python3
"""vLLM system benchmark: batch throughput with varying steering configs.

Measures total tokens/sec when processing batches with 0/1/4/8
distinct steering configurations.
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

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def run_throughput(
    model: str,
    num_prompts: int,
    prompt_len: int,
    max_tokens: int,
    distinct_configs: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
) -> dict:
    """Run throughput benchmark for a given config count."""
    from vllm import LLM, SamplingParams

    enable_steering = distinct_configs > 0
    max_steering = max(distinct_configs, 4) if enable_steering else 4

    print(f"    Loading model (steering={'on' if enable_steering else 'off'}, "
          f"max_configs={max_steering})...", flush=True)
    llm = LLM(
        model=model,
        enable_steering=enable_steering,
        max_steering_configs=max_steering,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompts = make_prompts(num_prompts, prompt_len)

    if distinct_configs == 0:
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        sp_list = [sp] * num_prompts
    else:
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_configs=distinct_configs,
            hook_points=["post_mlp"],
            scale=0.1,
            base_seed=42,
        )
        sp_list = []
        for i in range(num_prompts):
            sp = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                steering_vectors=diverse[i % distinct_configs],
            )
            sp_list.append(sp)

    # Warmup
    print(f"    Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    # Measure
    print(f"    Measuring ({iters} iters)...", flush=True)
    samples_ms = []
    total_output_tokens_list = []

    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        samples_ms.append(elapsed_ms)

        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_output_tokens_list.append(total_out)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Compute throughput
    avg_output_tokens = sum(total_output_tokens_list) / len(total_output_tokens_list)
    total_tokens_per_iter = num_prompts * prompt_len + avg_output_tokens  # input + output

    latency_stats = compute_stats(samples_ms)
    throughput_samples = [
        total_tokens_per_iter / (ms / 1000.0) for ms in samples_ms
    ]
    throughput_stats = compute_stats(throughput_samples)

    # Rename throughput stat keys from *_ms (misleading since the unit is
    # tokens/sec, not milliseconds) to *_tps.
    throughput_dict = {
        k.replace("_ms", "_tps"): v
        for k, v in throughput_stats.to_dict().items()
        if k != "samples_ms"
    }

    return {
        "latency_ms": {k: v for k, v in latency_stats.to_dict().items() if k != "samples_ms"},
        "throughput_tokens_per_sec": throughput_dict,
        "avg_output_tokens": avg_output_tokens,
        "total_tokens_per_iter": total_tokens_per_iter,
        "samples_ms": samples_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM throughput with steering")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--configs-sweep", default="0,1,4,8")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    config_counts = [int(x) for x in args.configs_sweep.split(",")]
    config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    print(f"Throughput benchmark: {args.model}")
    print(f"Prompts: {args.num_prompts}, prompt_len: {args.prompt_len}, max_tokens: {args.max_tokens}")
    print(f"Config counts to test: {config_counts}")
    print()

    all_results = []
    baseline_throughput = None

    for distinct_configs in config_counts:
        print(f"\n--- distinct_configs={distinct_configs} ---")

        try:
            result = run_throughput(
                model=args.model,
                num_prompts=args.num_prompts,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                distinct_configs=distinct_configs,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
            )

            mean_tps = result["throughput_tokens_per_sec"]["mean_tps"]
            mean_latency = result["latency_ms"]["mean_ms"]
            print(f"    throughput: {mean_tps:.0f} tokens/sec")
            print(f"    batch latency: {mean_latency:.0f} ms")

            if distinct_configs == 0:
                baseline_throughput = mean_tps

            overhead_pct = None
            if baseline_throughput and baseline_throughput > 0:
                overhead_pct = (baseline_throughput - mean_tps) / baseline_throughput * 100
                print(f"    throughput loss vs baseline: {overhead_pct:.1f}%")

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            result = {"error": "OOM"}
            overhead_pct = None

        params = {
            "model": args.model,
            "distinct_configs": distinct_configs,
            "num_prompts": args.num_prompts,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
        }
        results_out = {k: v for k, v in result.items() if k != "samples_ms"}
        if overhead_pct is not None:
            results_out["throughput_loss_pct"] = overhead_pct

        write_result(
            benchmark="vllm.throughput",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=result.get("samples_ms"),
        )
        all_results.append({
            "distinct_configs": distinct_configs,
            "results": results_out,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  Throughput Benchmark Summary: {args.model}")
    print(f"{'=' * 80}")
    print(f"{'configs':>10} {'tokens/sec':>14} {'batch_ms':>12} {'loss':>10}")
    print(f"{'-' * 80}")
    for r in all_results:
        res = r["results"]
        if "error" in res:
            print(f"{r['distinct_configs']:>10} {'OOM':>14}")
            continue
        tps = res["throughput_tokens_per_sec"]["mean_tps"]
        lat = res["latency_ms"]["mean_ms"]
        loss = res.get("throughput_loss_pct")
        loss_str = f"{loss:.1f}%" if loss is not None else "baseline"
        print(f"{r['distinct_configs']:>10} {tps:>14.0f} {lat:>12.0f} {loss_str:>10}")
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
