#!/usr/bin/env python3
"""Microbenchmark: raw steering op kernel latency.

Times torch.ops.vllm.apply_steering (or reference impl) in isolation
across hidden_size x num_tokens x table_rows x dtype.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import print_result_summary, write_result
from steering_bench.timing import cuda_timer

# Try to use the registered custom op; fall back to reference impl
try:
    import vllm  # noqa: F401 — triggers op registration

    def apply_steering(
        hidden_states: torch.Tensor,
        steering_table: torch.Tensor,
        steering_index: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.apply_steering(hidden_states, steering_table, steering_index)

    IMPL = "custom_op"
except ImportError:

    def apply_steering(
        hidden_states: torch.Tensor,
        steering_table: torch.Tensor,
        steering_index: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + steering_table[steering_index[: hidden_states.shape[0]]].to(
            hidden_states.dtype
        )

    IMPL = "reference"


FULL_SWEEP = {
    "hidden_size": [2048, 3072, 4096],
    "num_tokens": [1, 32, 128, 512, 2048],
    "num_table_rows": [4, 8, 16, 32],
    "dtype": [torch.float16, torch.bfloat16],
}

SUBSET_SWEEP = {
    "hidden_size": [2560],  # Gemma-3-4B
    "num_tokens": [1, 128, 2048],
    "num_table_rows": [4, 16],
    "dtype": [torch.bfloat16],
}


def run_single(
    hidden_size: int,
    num_tokens: int,
    num_table_rows: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: str,
) -> dict:
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    steering_table = torch.randn(num_table_rows, hidden_size, dtype=torch.float32, device=device)
    steering_index = torch.randint(0, num_table_rows, (num_tokens,), dtype=torch.long, device=device)

    stats = cuda_timer(warmup, iters, apply_steering, hidden_states, steering_table, steering_index)
    return stats.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Benchmark steering op kernel latency")
    parser.add_argument("--output-dir", default="results/micro/", help="Output directory")
    parser.add_argument("--subset", action="store_true", help="Run reduced sweep for quick testing")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    sweep = SUBSET_SWEEP if args.subset else FULL_SWEEP
    total = 1
    for v in sweep.values():
        total *= len(v)

    print(f"Steering op benchmark ({IMPL} implementation)")
    print(f"Device: {args.device}")
    print(f"Sweep: {total} configurations, {args.warmup} warmup, {args.iters} measured iters each")
    print()

    all_results = []
    idx = 0
    for hidden_size in sweep["hidden_size"]:
        for num_tokens in sweep["num_tokens"]:
            for num_table_rows in sweep["num_table_rows"]:
                for dtype in sweep["dtype"]:
                    idx += 1
                    dtype_str = str(dtype).split(".")[-1]
                    label = (
                        f"[{idx}/{total}] h={hidden_size} t={num_tokens} "
                        f"rows={num_table_rows} dtype={dtype_str}"
                    )
                    print(f"  {label} ...", end=" ", flush=True)

                    params = {
                        "hidden_size": hidden_size,
                        "num_tokens": num_tokens,
                        "num_table_rows": num_table_rows,
                        "dtype": dtype_str,
                        "implementation": IMPL,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    }

                    stats = run_single(
                        hidden_size, num_tokens, num_table_rows, dtype,
                        args.warmup, args.iters, args.device,
                    )
                    print(f"mean={stats['mean_ms']:.4f}ms median={stats['median_ms']:.4f}ms")

                    results = {
                        "latency_ms": {
                            k: v for k, v in stats.items() if k != "samples_ms"
                        },
                    }

                    path = write_result(
                        benchmark="micro.steering_op",
                        parameters=params,
                        results=results,
                        output_dir=args.output_dir,
                        tag=args.tag,
                        raw_samples_ms=stats["samples_ms"],
                    )
                    all_results.append({"params": params, "results": results, "path": str(path)})

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  Steering Op Benchmark Summary ({IMPL})")
    print(f"{'=' * 80}")
    print(f"{'hidden':>8} {'tokens':>8} {'rows':>6} {'dtype':>8} {'mean_ms':>10} {'p90_ms':>10} {'p99_ms':>10}")
    print(f"{'-' * 8:>8} {'-' * 8:>8} {'-' * 6:>6} {'-' * 8:>8} {'-' * 10:>10} {'-' * 10:>10} {'-' * 10:>10}")
    for r in all_results:
        p = r["params"]
        lat = r["results"]["latency_ms"]
        print(
            f"{p['hidden_size']:>8} {p['num_tokens']:>8} {p['num_table_rows']:>6} "
            f"{p['dtype']:>8} {lat['mean_ms']:>10.4f} {lat['p90_ms']:>10.4f} {lat['p99_ms']:>10.4f}"
        )
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
