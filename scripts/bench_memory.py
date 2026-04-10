#!/usr/bin/env python3
"""vLLM system benchmark: GPU memory cost of steering buffers.

Measures actual GPU memory delta for varying max_steering_configs
and compares against the theoretical formula.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import write_result

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}


def theoretical_memory_bytes(
    num_layers: int,
    hidden_size: int,
    max_steering_configs: int,
    max_batched_tokens: int = 8192,
) -> dict[str, int]:
    """Compute theoretical steering buffer memory cost.

    Per layer:
      3 hooks * (max_configs + 3) * hidden_size * 4 bytes (float32 tables)
    Shared:
      max_batched_tokens * 8 bytes (int64 steering_index, but shared across layers)
    """
    table_bytes_per_layer = 3 * (max_steering_configs + 3) * hidden_size * 4
    total_table_bytes = num_layers * table_bytes_per_layer
    index_bytes = max_batched_tokens * 8  # int64, shared across layers
    total = total_table_bytes + index_bytes

    return {
        "table_bytes_per_layer": table_bytes_per_layer,
        "total_table_bytes": total_table_bytes,
        "index_bytes": index_bytes,
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
    }


def _gpu_used_mb(device: int = 0) -> float:
    """Return total GPU memory used on *device* in MB.

    Uses torch.cuda.mem_get_info which queries the driver directly,
    so it sees memory held by other processes (including vLLM's
    EngineCore subprocess on v1). torch.cuda.memory_allocated()
    only sees the current process's tensors and returns 0 for
    subprocess-allocated memory.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / (1024 * 1024)


def _wait_for_gpu_settle(
    target_mb: float,
    tolerance_mb: float = 100.0,
    max_wait_sec: float = 30.0,
) -> None:
    """Poll until GPU used memory is within tolerance of target.

    After `del llm`, vLLM's subprocess may take a few seconds to
    release memory. This polls every 0.5s until we're within
    `tolerance_mb` of `target_mb` or `max_wait_sec` elapses.
    """
    import time
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        used = _gpu_used_mb()
        if abs(used - target_mb) <= tolerance_mb:
            return
        time.sleep(0.5)


def measure_model_memory(
    model: str,
    enable_steering: bool,
    max_steering_configs: int,
    baseline_mb: float,
    num_gpu_blocks_override: int,
    gpu_memory_utilization: float,
) -> dict[str, float]:
    """Load model, measure GPU memory, unload.

    Args:
        baseline_mb: GPU used-memory in MB before loading (from the
            very first call, representing the empty-GPU state).
        num_gpu_blocks_override: Force a fixed KV cache size so the
            delta between configs only reflects steering buffers,
            not variation in KV cache profiling.
        gpu_memory_utilization: Upper bound on vLLM's GPU memory use.

    Returns:
        dict with allocated_mb (model+buffers), delta_from_baseline_mb.
    """
    from vllm import LLM

    gc.collect()
    torch.cuda.empty_cache()

    mem_before = _gpu_used_mb()

    kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": 2048,
        "num_gpu_blocks_override": num_gpu_blocks_override,
    }
    if enable_steering:
        kwargs["enable_steering"] = True
        kwargs["max_steering_configs"] = max_steering_configs

    llm = LLM(**kwargs)

    mem_after_load = _gpu_used_mb()

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Wait for EngineCore subprocess to fully release memory
    _wait_for_gpu_settle(target_mb=baseline_mb, tolerance_mb=200.0)

    return {
        "allocated_mb": mem_after_load,
        "delta_from_baseline_mb": mem_after_load - baseline_mb,
        "delta_from_before_mb": mem_after_load - mem_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark steering buffer memory cost")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--configs-sweep", default="0,4,8,16,32")
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=256,
        help="Force a fixed KV cache size (in blocks). Smaller = cleaner "
             "steering delta measurement. Must be large enough for the model "
             "profiling step to succeed.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="Upper bound on vLLM's GPU memory usage. Lower leaves more "
             "headroom for the measurement.",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    config_counts = [int(x) for x in args.configs_sweep.split(",")]
    model_config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    # Capture empty-GPU baseline (before any model is loaded)
    gc.collect()
    torch.cuda.empty_cache()
    empty_gpu_mb = _gpu_used_mb()

    print(f"Memory benchmark: {args.model}")
    print(f"Config counts: {config_counts}")
    print(f"hidden_size={model_config['hidden_size']}, num_layers={model_config['num_layers']}")
    print(f"Empty-GPU baseline: {empty_gpu_mb:.1f} MB (other processes on device)")
    print(f"Fixed KV blocks: {args.num_gpu_blocks}, gpu_util: {args.gpu_memory_utilization}")
    print()

    all_results = []
    baseline_allocated = None

    for max_configs in config_counts:
        enable = max_configs > 0
        label = f"max_configs={max_configs}" if enable else "steering_off"
        print(f"--- {label} ---")

        print(f"  Loading model...", flush=True)
        try:
            mem = measure_model_memory(
                args.model,
                enable,
                max_configs,
                baseline_mb=empty_gpu_mb,
                num_gpu_blocks_override=args.num_gpu_blocks,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM!")
            all_results.append({
                "max_configs": max_configs,
                "results": {"error": "OOM"},
            })
            continue

        print(
            f"  allocated: {mem['allocated_mb']:.1f} MB "
            f"(delta from empty: {mem['delta_from_baseline_mb']:.1f} MB)"
        )

        if max_configs == 0:
            baseline_allocated = mem["allocated_mb"]

        steering_delta_mb = None
        if baseline_allocated is not None:
            steering_delta_mb = mem["allocated_mb"] - baseline_allocated
            print(f"  steering delta vs configs=0: {steering_delta_mb:.1f} MB")

        # Theoretical
        if max_configs > 0:
            theory = theoretical_memory_bytes(
                num_layers=model_config["num_layers"],
                hidden_size=model_config["hidden_size"],
                max_steering_configs=max_configs,
            )
            print(f"  theoretical: {theory['total_mb']:.1f} MB")
        else:
            theory = None

        params = {
            "model": args.model,
            "enable_steering": enable,
            "max_steering_configs": max_configs,
        }
        results = {
            "allocated_mb": mem["allocated_mb"],
            "delta_from_baseline_mb": mem["delta_from_baseline_mb"],
        }
        if steering_delta_mb is not None:
            results["steering_delta_mb"] = steering_delta_mb
        if theory is not None:
            results["theoretical_mb"] = theory["total_mb"]
            results["theoretical_detail"] = theory

        write_result(
            benchmark="vllm.memory",
            parameters=params,
            results=results,
            output_dir=args.output_dir,
            tag=args.tag,
        )
        all_results.append({
            "max_configs": max_configs,
            "results": results,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  Memory Benchmark Summary: {args.model}")
    print(f"{'=' * 80}")
    print(f"{'configs':>10} {'allocated':>12} {'delta':>10} {'theoretical':>14} {'ratio':>8}")
    print(f"{'-' * 80}")
    for r in all_results:
        res = r["results"]
        if "error" in res:
            print(f"{r['max_configs']:>10} {'OOM':>12}")
            continue
        delta = res.get("steering_delta_mb")
        theory = res.get("theoretical_mb")
        delta_str = f"{delta:.1f} MB" if delta is not None else "baseline"
        theory_str = f"{theory:.1f} MB" if theory is not None else "N/A"
        ratio_str = ""
        if delta is not None and theory is not None and theory > 0:
            ratio_str = f"{delta / theory:.2f}x"
        print(
            f"{r['max_configs']:>10} {res['allocated_mb']:>10.1f} MB "
            f"{delta_str:>10} {theory_str:>14} {ratio_str:>8}"
        )
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
