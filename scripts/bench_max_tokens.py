#!/usr/bin/env python3
"""Max-tokens sweep benchmark: shows how per-step overhead converges to
the populate floor as max_tokens grows.

Hypothesis: per-iter overhead has two components,
    total_overhead_per_iter ≈ populate_per_step * max_tokens
                              + submission_per_active * num_active

So per-step overhead is:
    total_per_step ≈ populate_per_step + submission_per_active * num_active / max_tokens

The submission term decays as 1/max_tokens. At small max_tokens it dominates;
at large max_tokens it's negligible and we see the populate floor.

Each sweep point reuses a single model load so the comparison is clean.
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
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def measure_one(llm, prompts, sp_list, warmup, iters):
    """Warmup + measured iters around llm.generate()."""
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    samples_ms: list[float] = []
    output_tokens_per_iter: list[int] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
        output_tokens_per_iter.append(
            sum(len(o.outputs[0].token_ids) for o in outputs)
        )

    latency_stats = compute_stats(samples_ms).to_dict()
    avg_output_tokens = sum(output_tokens_per_iter) / len(output_tokens_per_iter)
    return {
        "latency": {k: v for k, v in latency_stats.items() if k != "samples_ms"},
        "avg_output_tokens": avg_output_tokens,
    }


def build_sp_list(
    batch_size,
    num_active,
    max_tokens,
    steering_vectors_list,
    sampling_params_cls,
):
    """Build a per-request SamplingParams list.

    ``steering_vectors_list`` is a list of exactly ``num_active`` steering
    vector dicts. When all entries point to the same dict object (shared
    mode) the steered slots share one SamplingParams instance. When each
    entry is a distinct dict (distinct mode) each steered slot gets its
    own SamplingParams — which means each one also gets its own unique
    ``(config_hash, phase)`` key and hits the slow register_config path.
    """
    sp_unsteered = sampling_params_cls(max_tokens=max_tokens, temperature=0.0)

    if num_active == 0 or steering_vectors_list is None:
        return [sp_unsteered] * batch_size

    # Detect shared vs distinct: if all entries are the same object id, we
    # can reuse one SamplingParams and get @cached_property benefits.
    first = steering_vectors_list[0]
    all_same = all(v is first for v in steering_vectors_list)

    if all_same:
        sp_steered = sampling_params_cls(
            max_tokens=max_tokens,
            temperature=0.0,
            steering_vectors=first,
        )
        steered_sps = [sp_steered] * num_active
    else:
        # Distinct mode: one fresh SamplingParams per active slot.
        steered_sps = [
            sampling_params_cls(
                max_tokens=max_tokens,
                temperature=0.0,
                steering_vectors=steering_vectors_list[i],
            )
            for i in range(num_active)
        ]

    return steered_sps + [sp_unsteered] * (batch_size - num_active)


def main():
    parser = argparse.ArgumentParser(description="Max-tokens sweep")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--max-tokens-list",
        default="64,128,256,512,1024,2048",
        help="Comma-separated list of max_tokens values to sweep.",
    )
    parser.add_argument(
        "--num-active-list",
        default="0,1,8,16",
        help=(
            "Comma-separated list of num_active values. 0 = all unsteered "
            "(populate not running), used as the disabled/idle baseline."
        ),
    )
    parser.add_argument("--max-steering-configs", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--distinct-vectors",
        action="store_true",
        help=(
            "Give each steered slot its own unique random vector (and its "
            "own SamplingParams instance). Default behavior is shared-vector "
            "mode where all active slots reuse one vector/SP — which lets "
            "hash caching amortize across the batch. Distinct mode exercises "
            "the slow register_config new-row path for every slot and is a "
            "more realistic worst case for personalization-style workloads."
        ),
    )
    args = parser.parse_args()

    max_tokens_list = [int(x) for x in args.max_tokens_list.split(",")]
    num_active_list = [int(x) for x in args.num_active_list.split(",")]

    for n in num_active_list:
        if n > args.batch_size:
            parser.error(
                f"--num-active-list contains {n} > --batch-size {args.batch_size}"
            )

    model_config = MODEL_CONFIGS.get(
        args.model, {"hidden_size": 2560, "num_layers": 34}
    )

    from vllm import LLM, SamplingParams

    print(
        f"Loading model {args.model} (steering enabled, "
        f"max_steering_configs={args.max_steering_configs})..."
    )
    llm = LLM(
        model=args.model,
        enable_steering=True,
        max_steering_configs=args.max_steering_configs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(2048, max(max_tokens_list) + args.prompt_len + 64),
    )

    # Build the pool of steering vectors. Shared mode: one vector, shared by
    # every active slot. Distinct mode: one vector per possible active slot,
    # each with its own seed, so every active request gets a unique config.
    max_active = max(num_active_list) if num_active_list else 0
    if args.distinct_vectors and max_active > 0:
        vector_pool = [
            random_steering_vectors(
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                hook_points=["post_mlp"],
                scale=0.1,
                seed=42 + i,
            )
            for i in range(max_active)
        ]
    else:
        shared = random_steering_vectors(
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            hook_points=["post_mlp"],
            scale=0.1,
            seed=42,
        )
        # Same object reused everywhere — build_sp_list detects this and
        # creates a single shared SamplingParams.
        vector_pool = [shared] * max(1, max_active)

    prompts = make_prompts(args.batch_size, args.prompt_len)

    mode_label = "distinct vectors" if args.distinct_vectors else "shared vector"
    print(
        f"\nSweeping max_tokens={max_tokens_list} x num_active={num_active_list} "
        f"at batch_size={args.batch_size} ({mode_label})..."
    )

    for max_tokens in max_tokens_list:
        print(f"\n  max_tokens={max_tokens}")
        for num_active in num_active_list:
            # Slice the pool to num_active entries for this measurement.
            slot_vectors = vector_pool[:num_active] if num_active > 0 else None
            sp_list = build_sp_list(
                args.batch_size,
                num_active,
                max_tokens,
                slot_vectors,
                SamplingParams,
            )

            try:
                result = measure_one(
                    llm, prompts, sp_list, args.warmup, args.iters
                )
                lat_mean = result["latency"]["mean_ms"]
                avg_out = result["avg_output_tokens"]
                per_step = lat_mean / max(1, avg_out)
                print(
                    f"    num_active={num_active:>3} → "
                    f"latency={lat_mean:.1f} ms ({avg_out:.0f} out tok), "
                    f"per_step={per_step:.3f} ms/step"
                )

                if num_active == 0:
                    num_distinct = 0
                elif args.distinct_vectors:
                    num_distinct = num_active
                else:
                    num_distinct = 1
                params = {
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "num_active": num_active,
                    "num_distinct_configs": num_distinct,
                    "shared_vector": not args.distinct_vectors,
                    "max_tokens": max_tokens,
                    "prompt_len": args.prompt_len,
                    "max_steering_configs": args.max_steering_configs,
                    "enable_steering": True,
                }
                write_result(
                    benchmark="vllm.max_tokens_sweep",
                    parameters=params,
                    results={
                        "latency_ms": result["latency"],
                        "avg_output_tokens": result["avg_output_tokens"],
                        "per_step_ms": per_step,
                    },
                    output_dir=args.output_dir,
                    tag=args.tag,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"    num_active={num_active:>3} → OOM at "
                    f"max_tokens={max_tokens}, skipping"
                )
                gc.collect()
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
