#!/usr/bin/env python3
"""Minimal generate() wrapper for nsys profiling.

No torch.profiler inside — avoids CUPTI_ERROR_MULTIPLE_SUBSCRIBERS
when nsys wraps this process. Designed to be called under nsys profile.

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile \\
      --output=trace --force-overwrite=true --trace=cuda,nvtx,osrt \\
      python scripts/nsys_target.py --mode steering --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steering_bench.vectors import random_steering_vectors

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
}


def main():
    parser = argparse.ArgumentParser(description="Minimal nsys profiling target")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--mode", choices=["disabled", "steering"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-active",
        type=int,
        default=-1,
        help=(
            "Number of steered requests in the batch (only used when mode=steering). "
            "By default each active request gets its own distinct random vector "
            "(num_active distinct configs). Pass --shared-vector to make all "
            "active slots share one vector (1 distinct config). "
            "-1 (default) means all batch slots are steered."
        ),
    )
    parser.add_argument(
        "--shared-vector",
        action="store_true",
        help=(
            "All --num-active slots share one steering vector (1 distinct config) "
            "instead of each getting its own (num_active distinct configs). "
            "Use this to isolate populate's per-distinct-config cost from any "
            "per-active engine work."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})
    enable_steering = args.mode == "steering"

    num_active = args.num_active if args.num_active >= 0 else args.batch_size
    if num_active > args.batch_size:
        parser.error(f"--num-active ({num_active}) > --batch-size ({args.batch_size})")
    if not enable_steering:
        num_active = 0

    num_distinct_configs = 1 if args.shared_vector else num_active

    # Need one table row per distinct active config plus headroom for the
    # global rows (rows 0–2 are reserved for sentinel/global vectors).
    # Always size to max(4, num_active) so that --shared-vector and the
    # default mode use the same table size at a given num_active — keeps
    # table size from being a confounding variable in the comparison.
    max_steering_configs = max(4, num_active)

    print(
        f"Loading model (mode={args.mode}, batch_size={args.batch_size}, "
        f"num_active={num_active}, num_distinct_configs={num_distinct_configs}, "
        f"max_steering_configs={max_steering_configs}, "
        f"gpu_util={args.gpu_memory_utilization})..."
    )
    llm = LLM(
        model=args.model,
        enable_steering=enable_steering,
        max_steering_configs=max_steering_configs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )

    words = " ".join(["hello"] * max(1, int(args.prompt_len / 1.3)))
    prompts = [words] * args.batch_size

    if enable_steering and num_active > 0:
        plain_sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        if args.shared_vector:
            # All active slots share one vector → 1 distinct config.
            vectors = random_steering_vectors(
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                hook_points=["post_mlp"],
                scale=0.1,
                seed=42,
            )
            shared_sp = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                steering_vectors=vectors,
            )
            sp_list = [shared_sp] * num_active + [plain_sp] * (
                args.batch_size - num_active
            )
        else:
            # Each active slot gets a distinct vector → num_active distinct configs.
            steered_sps: list[SamplingParams] = []
            for i in range(num_active):
                vectors = random_steering_vectors(
                    hidden_size=model_config["hidden_size"],
                    num_layers=model_config["num_layers"],
                    hook_points=["post_mlp"],
                    scale=0.1,
                    seed=42 + i,
                )
                steered_sps.append(
                    SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        steering_vectors=vectors,
                    )
                )
            sp_list = steered_sps + [plain_sp] * (args.batch_size - num_active)
    else:
        plain_sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        sp_list = [plain_sp] * args.batch_size

    import torch

    # Warmup (not profiled by nsys default — full trace captures everything,
    # but the warmup iters are distinguishable in the timeline by their
    # compilation / graph-capture activity)
    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        llm.generate(prompts, sp_list)

    # Measured iters — these are the ones to look at in the nsys timeline
    print(f"Generating ({args.iters} iters, batch={args.batch_size}, "
          f"num_active={num_active}, num_distinct_configs={num_distinct_configs}, "
          f"max_tokens={args.max_tokens})...")
    iter_times_ms: list[float] = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        iter_ms = (t1 - t0) * 1000.0
        iter_times_ms.append(iter_ms)
        print(f"  iter {i + 1}/{args.iters} done in {iter_ms:.1f} ms")

    if iter_times_ms:
        mean_ms = sum(iter_times_ms) / len(iter_times_ms)
        # Sort and take the median to mitigate prefix-cache or graph-capture noise
        sorted_ms = sorted(iter_times_ms)
        median_ms = sorted_ms[len(sorted_ms) // 2]
        per_step_ms = mean_ms / args.max_tokens
        print(
            f"\nPer-iter wall clock: mean={mean_ms:.1f} ms, "
            f"median={median_ms:.1f} ms, per-step={per_step_ms:.3f} ms "
            f"(over {args.max_tokens} max_tokens)"
        )

    print("Done.")


if __name__ == "__main__":
    main()
