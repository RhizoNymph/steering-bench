#!/usr/bin/env python3
"""Profile steering overhead in a full model forward pass.

Runs vLLM generate() with torch.profiler active, once with steering
disabled and once with steering enabled at configs=1. Saves Chrome
traces and prints ranked CPU/CUDA operation tables so you can see
exactly where the time is going.

IMPORTANT: vLLM v1 runs the model in an EngineCore subprocess by
default. torch.profiler in the main process won't see GPU kernels
that happen in the subprocess. This script sets
VLLM_ENABLE_V1_MULTIPROCESSING=0 to force in-process execution so
the profiler can capture the full picture.

If the in-process mode doesn't work for your vLLM version, fall
back to Nsight Systems:
    nsys profile --stats=true -o trace uv run --no-sync python \\
        scripts/bench_latency.py --batch-sizes 8 --iters 3 --warmup 1
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# CRITICAL: must be set before importing vllm
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.profiler import ProfilerActivity, profile

from steering_bench.vectors import random_steering_vectors

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def build_sp_list(
    mode: str,
    batch_size: int,
    max_tokens: int,
    hidden_size: int,
    num_layers: int,
):
    from vllm import SamplingParams

    if mode == "disabled":
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        return [sp] * batch_size

    # mode == "steering"
    vectors = random_steering_vectors(
        hidden_size=hidden_size,
        num_layers=num_layers,
        hook_points=["post_mlp"],
        scale=0.1,
        seed=42,
    )
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        steering_vectors=vectors,
    )
    return [sp] * batch_size


def profile_mode(
    llm,
    mode: str,
    prompts: list[str],
    sp_list,
    warmup: int,
    iters: int,
    output_dir: Path,
) -> dict:
    """Profile one mode and return a dict of top events + trace path."""
    print(f"\n{'=' * 70}")
    print(f"  Profiling mode: {mode}")
    print(f"{'=' * 70}")

    # Warmup
    print(f"  Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"  Profiling ({iters} iters)...", flush=True)
    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iters):
            llm.generate(prompts, sp_list)

    # Save Chrome trace
    trace_path = output_dir / f"trace_{mode}.json.gz"
    prof.export_chrome_trace(str(trace_path))
    print(f"  Chrome trace: {trace_path}")

    # Detect if the profiler saw any CUDA activity
    averages = list(prof.key_averages())
    cuda_total = sum(a.cuda_time_total for a in averages) / 1000.0  # ms
    cpu_total = sum(a.self_cpu_time_total for a in averages) / 1000.0  # ms
    print(f"  Total captured: cpu={cpu_total:.1f}ms cuda={cuda_total:.1f}ms")

    if cuda_total == 0:
        print(
            "  WARNING: zero CUDA time captured. The profiler may not see "
            "the vLLM worker subprocess. Check that VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "was respected, or use Nsight Systems instead (see script docstring)."
        )

    # Top CUDA events
    print(f"\n  Top 20 CUDA events (by cuda_time_total):")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    # Top CPU events
    print(f"\n  Top 20 CPU events (by self_cpu_time_total):")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20,
        )
    )

    # Steering-specific filter
    print(f"\n  Steering-related events:")
    steering_events = []
    for avg in averages:
        name = avg.key.lower()
        if any(term in name for term in ["steering", "apply_steering", "populate"]):
            steering_events.append(avg)

    if steering_events:
        for avg in sorted(steering_events, key=lambda a: a.cuda_time_total, reverse=True):
            cpu_ms = avg.self_cpu_time_total / 1000.0
            cuda_ms = avg.cuda_time_total / 1000.0
            print(
                f"    {avg.key[:60]:<60} "
                f"cpu={cpu_ms:>8.2f}ms cuda={cuda_ms:>8.2f}ms count={avg.count:>6}"
            )
    else:
        print("    (none found)")

    return {
        "trace_path": str(trace_path),
        "cpu_total_ms": cpu_total,
        "cuda_total_ms": cuda_total,
        "num_events": len(averages),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile steering overhead")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/profile/")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Measured iterations. Keep small — profiling is slow.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Reduce for faster profiling. Default 64 for speed.",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify the multiprocessing override
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        print(
            "WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not '0'. "
            "The profiler may not capture worker GPU activity."
        )

    prompts = make_prompts(args.batch_size, args.prompt_len)

    # Profile each mode
    summary: dict[str, dict] = {}

    for mode in ["disabled", "steering"]:
        from vllm import LLM

        enable_steering = mode != "disabled"

        print(f"\n[{mode}] Loading model (enable_steering={enable_steering})...", flush=True)
        llm = LLM(
            model=args.model,
            enable_steering=enable_steering,
            max_steering_configs=4,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )

        sp_list = build_sp_list(
            mode=mode,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
        )

        summary[mode] = profile_mode(
            llm=llm,
            mode=mode,
            prompts=prompts,
            sp_list=sp_list,
            warmup=args.warmup,
            iters=args.iters,
            output_dir=output_dir,
        )

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'=' * 70}")
    print("  Final Summary")
    print(f"{'=' * 70}")
    print(f"{'mode':<12} {'cpu_ms':>12} {'cuda_ms':>12} {'num_events':>12}")
    print(f"{'-' * 70}")
    for mode, data in summary.items():
        print(
            f"{mode:<12} {data['cpu_total_ms']:>12.1f} "
            f"{data['cuda_total_ms']:>12.1f} {data['num_events']:>12}"
        )
    print(f"{'=' * 70}")

    if "disabled" in summary and "steering" in summary:
        d = summary["disabled"]
        s = summary["steering"]
        cpu_delta = s["cpu_total_ms"] - d["cpu_total_ms"]
        cuda_delta = s["cuda_total_ms"] - d["cuda_total_ms"]
        print(f"\n  Steering overhead:")
        print(f"    CPU delta:  {cpu_delta:+.1f} ms")
        print(f"    CUDA delta: {cuda_delta:+.1f} ms")
        if d["cuda_total_ms"] > 0:
            print(
                f"    CUDA relative: {cuda_delta / d['cuda_total_ms'] * 100:+.1f}% "
                f"(over {args.iters} iters)"
            )

    print(f"\nOpen Chrome traces at chrome://tracing or https://ui.perfetto.dev")
    print(f"Traces saved to: {output_dir}")


if __name__ == "__main__":
    main()
