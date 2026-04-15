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
    distinct_vectors: bool = False,
    max_steering_configs: int = 4,
    gpu_memory_utilization: float = 0.9,
) -> dict:
    """Run a mixed batch: num_active of batch_size requests are steered.

    When ``distinct_vectors`` is False (default) all active slots share
    one steering vector and one SamplingParams — the common global-
    steering-target case where hash caching amortizes across the batch.
    When True, each active slot gets its own unique vector and its own
    fresh SamplingParams — the personalization-style case where every
    request hits the slow register_config new-row path.
    """
    from vllm import LLM, SamplingParams

    print(
        f"    Loading model (batch={batch_size}, active={num_active}/{batch_size}, "
        f"distinct={distinct_vectors})...",
        flush=True,
    )
    llm = LLM(
        model=model,
        enable_steering=True,
        max_steering_configs=max_steering_configs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )

    prompts = make_prompts(batch_size, prompt_len)

    # Non-steered sampling params (no steering_vectors field)
    sp_unsteered = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Build the sp_list. Shared mode: one shared steered SP. Distinct mode:
    # each active slot gets its own SP with a unique vector.
    sp_list: list = []
    if num_active > 0 and distinct_vectors:
        for i in range(num_active):
            v = random_steering_vectors(
                hidden_size=hidden_size,
                num_layers=num_layers,
                hook_points=["post_mlp"],
                scale=0.1,
                seed=42 + i,
            )
            sp_list.append(
                SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.0,
                    steering_vectors=v,
                )
            )
    elif num_active > 0:
        shared_vectors = random_steering_vectors(
            hidden_size=hidden_size,
            num_layers=num_layers,
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        sp_steered = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            steering_vectors=shared_vectors,
        )
        sp_list = [sp_steered] * num_active

    sp_list.extend([sp_unsteered] * (batch_size - num_active))

    # Warmup
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    # Measure
    samples_ms = []
    total_output_tokens_per_iter = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
        total_output_tokens_per_iter.append(
            sum(len(o.outputs[0].token_ids) for o in outputs)
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    latency_stats = compute_stats(samples_ms).to_dict()

    # Throughput computation
    avg_output_tokens = (
        sum(total_output_tokens_per_iter) / len(total_output_tokens_per_iter)
        if total_output_tokens_per_iter
        else 0
    )
    total_tokens_per_iter = batch_size * prompt_len + avg_output_tokens
    throughput_samples = [
        total_tokens_per_iter / (ms / 1000.0) for ms in samples_ms
    ]
    throughput_stats = {
        k.replace("_ms", "_tps"): v
        for k, v in compute_stats(throughput_samples).to_dict().items()
        if k != "samples_ms"
    }

    return {
        "latency": latency_stats,
        "throughput": throughput_stats,
        "avg_output_tokens_total": avg_output_tokens,
    }


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
    parser.add_argument(
        "--distinct-vectors",
        action="store_true",
        help=(
            "Give each active slot a distinct steering vector (and a "
            "distinct SamplingParams instance). Default is shared-vector."
        ),
    )
    parser.add_argument(
        "--max-steering-configs",
        type=int,
        default=None,
        help=(
            "Override max_steering_configs. Default: 4 for shared-vector "
            "runs, max(batch_size, 4) for distinct-vector runs so the "
            "scheduler doesn't serialize on table capacity."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=(
            "Passed through to LLM(). The script does N model loads in one "
            "Python process (one per --num-active value), and the in-process "
            "CUDA allocator does not reliably return GPU memory to the OS "
            "between loads. Lower this (e.g. 0.7) if you OOM partway through."
        ),
    )
    parser.add_argument(
        "--num-active-only",
        type=int,
        default=None,
        help=(
            "Run only one num_active cell instead of the full sweep. Use "
            "this to put exactly one model load per Python process when the "
            "in-process CUDA allocator leak is bad enough that even lower "
            "--gpu-memory-utilization isn't enough."
        ),
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    bs = args.batch_size
    if args.num_active_only is not None:
        if args.num_active_only > bs:
            parser.error(
                f"--num-active-only={args.num_active_only} > --batch-size={bs}"
            )
        active_counts = [args.num_active_only]
    else:
        active_counts = [0, 1, max(1, bs // 4), max(1, bs // 2), bs]
        # Deduplicate (e.g. batch_size=4 collapses)
        active_counts = sorted(set(active_counts))
    labels = {0: "none_active", bs: "all_active"}

    # In distinct-vector mode each active slot needs its own table row.
    # Size the table to at least batch_size so we never hit capacity.
    if args.max_steering_configs is not None:
        max_steering_configs = args.max_steering_configs
    elif args.distinct_vectors:
        max_steering_configs = max(bs, 4)
    else:
        max_steering_configs = 4

    mode = "distinct-vector" if args.distinct_vectors else "shared-vector"
    print(f"Mixed-batch benchmark: {args.model}")
    print(f"Mode: {mode}")
    print(f"Batch size: {bs}")
    print(f"max_steering_configs: {max_steering_configs}")
    print(f"Active counts tested: {active_counts}")
    print()

    all_results = []
    baseline_latency = None
    baseline_tps = None

    for num_active in active_counts:
        label = labels.get(num_active, f"{num_active}_active")
        print(f"--- {label} ({num_active}/{bs} active) ---")

        result = None
        latency_overhead_pct = None
        throughput_loss_pct = None

        try:
            result = run_mixed(
                model=args.model,
                batch_size=bs,
                num_active=num_active,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                distinct_vectors=args.distinct_vectors,
                max_steering_configs=max_steering_configs,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            mean_ms = result["latency"]["mean_ms"]
            p90_ms = result["latency"]["p90_ms"]
            mean_tps = result["throughput"]["mean_tps"]

            if num_active == 0:
                baseline_latency = mean_ms
                baseline_tps = mean_tps

            if baseline_latency and baseline_latency > 0:
                latency_overhead_pct = (mean_ms - baseline_latency) / baseline_latency * 100
            if baseline_tps and baseline_tps > 0:
                throughput_loss_pct = (baseline_tps - mean_tps) / baseline_tps * 100

            print(
                f"    mean={mean_ms:.1f}ms p90={p90_ms:.1f}ms  "
                f"throughput={mean_tps:.0f} tok/s  "
                f"lat_overhead={latency_overhead_pct:+.1f}% "
                f"tps_loss={throughput_loss_pct:+.1f}%"
                if latency_overhead_pct is not None
                else f"    mean={mean_ms:.1f}ms throughput={mean_tps:.0f} tok/s (baseline)"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            result = {"error": "OOM"}

        if num_active == 0:
            num_distinct = 0
        elif args.distinct_vectors:
            num_distinct = num_active
        else:
            num_distinct = 1

        params = {
            "model": args.model,
            "batch_size": bs,
            "num_active": num_active,
            "active_fraction": num_active / bs if bs > 0 else 0,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "distinct_vectors": args.distinct_vectors,
            "num_distinct_configs": num_distinct,
            "max_steering_configs": max_steering_configs,
        }
        results_out: dict = {}
        if result and "error" not in result:
            results_out["latency_ms"] = {
                k: v for k, v in result["latency"].items() if k != "samples_ms"
            }
            results_out["throughput_tokens_per_sec"] = result["throughput"]
            results_out["avg_output_tokens_total"] = result["avg_output_tokens_total"]
        else:
            results_out["error"] = "OOM"
        if latency_overhead_pct is not None:
            results_out["latency_overhead_pct"] = latency_overhead_pct
        if throughput_loss_pct is not None:
            results_out["throughput_loss_pct"] = throughput_loss_pct

        write_result(
            benchmark="vllm.mixed_batch",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=result["latency"].get("samples_ms") if result and "error" not in result else None,
        )
        all_results.append({
            "label": label,
            "num_active": num_active,
            "result": result,
            "latency_overhead_pct": latency_overhead_pct,
            "throughput_loss_pct": throughput_loss_pct,
        })

    print(f"\n{'=' * 100}")
    print(f"  Mixed-Batch Summary (batch={bs}, {args.model})")
    print(f"{'=' * 100}")
    print(
        f"{'mode':<16} {'active':>10} {'mean_ms':>10} {'p90_ms':>10} "
        f"{'tok/sec':>12} {'lat_over':>12} {'tps_loss':>12}"
    )
    print(f"{'-' * 100}")
    for r in all_results:
        result = r["result"]
        if not result or "error" in result:
            print(f"{r['label']:<16} {r['num_active']:>10} {'OOM':>10}")
            continue
        lat = result["latency"]
        tps = result["throughput"]["mean_tps"]
        lat_over = r["latency_overhead_pct"]
        tps_loss = r["throughput_loss_pct"]
        lat_over_str = f"{lat_over:+.1f}%" if lat_over is not None else "baseline"
        tps_loss_str = f"{tps_loss:+.1f}%" if tps_loss is not None else "baseline"
        print(
            f"{r['label']:<16} {r['num_active']:>10} "
            f"{lat['mean_ms']:>10.1f} {lat['p90_ms']:>10.1f} "
            f"{tps:>12.0f} {lat_over_str:>12} {tps_loss_str:>12}"
        )
    print(f"{'=' * 100}")
    print(
        "\nInterpretation:\n"
        "  - If 'one_active' latency/throughput is close to 'all_active', the cost is\n"
        "    TRANSITIVE: any active request forces the full per-request cost on the\n"
        "    entire batch, including non-steered requests sharing the step.\n"
        "  - If 'one_active' scales with active count toward 'all_active', the cost\n"
        "    is PROPORTIONAL to the number of steered requests.\n"
        "  - In vLLM's continuous batching, the theoretical expectation is TRANSITIVE\n"
        "    cost because the forward pass is shared across all tokens in a step.\n"
    )


if __name__ == "__main__":
    main()
