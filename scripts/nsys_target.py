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
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})
    enable_steering = args.mode == "steering"

    print(f"Loading model (mode={args.mode}, gpu_util={args.gpu_memory_utilization})...")
    llm = LLM(
        model=args.model,
        enable_steering=enable_steering,
        max_steering_configs=4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )

    words = " ".join(["hello"] * max(1, int(args.prompt_len / 1.3)))
    prompts = [words] * args.batch_size

    if enable_steering:
        vectors = random_steering_vectors(
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        sp = SamplingParams(
            max_tokens=args.max_tokens, temperature=0.0, steering_vectors=vectors,
        )
    else:
        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    sp_list = [sp] * args.batch_size

    # Warmup (not profiled by nsys default — full trace captures everything,
    # but the warmup iters are distinguishable in the timeline by their
    # compilation / graph-capture activity)
    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        llm.generate(prompts, sp_list)

    # Measured iters — these are the ones to look at in the nsys timeline
    print(f"Generating ({args.iters} iters, batch={args.batch_size}, "
          f"max_tokens={args.max_tokens})...")
    for i in range(args.iters):
        llm.generate(prompts, sp_list)
        print(f"  iter {i + 1}/{args.iters} done")

    print("Done.")


if __name__ == "__main__":
    main()
