#!/usr/bin/env python3
"""Capture manager microbenchmark: plan building, GPU gather, dispatch.

Measures the three phases of the CaptureManager hot path in isolation,
using real CUDA tensors and NullCaptureSink (no I/O) to isolate
manager overhead from disk/network.

Sweep dimensions:
  batch_size        — requests per step (1, 8, 32)
  num_consumers     — NullCaptureSinks in parallel (1, 2, 4, 8)
  position_type     — last_prompt (1 row) or all_prompt (prompt_len rows)
  num_layers        — decoder layers captured (1, 6, 12)

Three separately reported phases:
  build_ms  — build_step_plan() (CPU, O(requests × consumers × layers × positions))
  hook_ms   — on_hook() × num_layers (GPU index_select)
  dispatch_ms — dispatch_step_captures() (CPU fan-out + GPU→CPU copy)
"""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.capture_consumers.consumers import NullCaptureSink
from steering_bench.output import print_result_summary, write_result
from steering_bench.timing import compute_stats

# ──────────────────────────────────────────────────────────── model configs

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "facebook/opt-125m": {"hidden_size": 768, "num_hidden_layers": 12},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_hidden_layers": 16},
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_hidden_layers": 34},
}


# ──────────────────────────────────────────────────────────── helpers

def _make_batch_view(
    batch_size: int,
    prompt_len: int,
    req_ids: list[str],
):
    """Simulate a prefill step where all prompt tokens are scheduled."""
    from vllm.v1.capture.plan import CaptureBatchView

    return CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=[prompt_len] * batch_size,
        num_computed_tokens=[0] * batch_size,
        num_scheduled_tokens=[prompt_len] * batch_size,
        token_offsets=[i * prompt_len for i in range(batch_size)],
    )


def _make_manager(
    num_consumers: int,
    num_hidden_layers: int,
    hidden_size: int,
    layers_to_capture: list[int],
    hook_names: list[str],
    position_type: str,
    device: torch.device,
):
    """Build a CaptureManager with NullCaptureSinks on CUDA.

    ``hook_names`` is a list — each entry becomes a key in the
    ``CaptureSpec.hooks`` dict, mapped to the same ``layers_to_capture``.
    Use a single-element list for a single-hook benchmark.
    """
    from vllm.v1.capture.manager import CaptureManager
    from vllm.v1.capture.types import CaptureSpec

    spec = CaptureSpec(
        hooks={name: layers_to_capture for name in hook_names},
        positions=position_type,
    )

    sinks = tuple(NullCaptureSink() for _ in range(num_consumers))
    specs = tuple(spec for _ in range(num_consumers))

    return CaptureManager(
        consumers=sinks,
        consumer_specs=specs,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        model_dtype=torch.float16,
        device=device,
    )


def _run_one(
    manager,
    batch_view,
    req_ids: list[str],
    hidden: torch.Tensor,
    hook_names: list[str],
    layers_to_capture: list[int],
    warmup: int,
    iters: int,
) -> dict[str, list[float]]:
    """Run warmup + measurement and return per-phase sample lists.

    ``hook_names`` drives the on_hook inner loop — for each (layer, hook)
    pair in the product, on_hook fires once per iteration.  The hook
    phase timing covers the entire sequence of kernels.
    """
    build_samples: list[float] = []
    hook_samples: list[float] = []
    dispatch_samples: list[float] = []

    prompt_len = batch_view.num_prompt_tokens[0]

    for phase in ("warmup", "measure"):
        n = warmup if phase == "warmup" else iters
        for _ in range(n):
            # Register requests.
            for rid in req_ids:
                manager.register_request(
                    req_id=rid,
                    client_specs=None,
                    num_prompt_tokens=prompt_len,
                )

            # ── Phase 1: build_step_plan (CPU)
            t0 = time.perf_counter()
            plan = manager.build_step_plan(batch_view)
            t1 = time.perf_counter()

            # ── Phase 2: on_hook (GPU index_select)
            # hook_ms = GPU kernel execution time only. The GPU→CPU copy of
            # sliced activations happens in dispatch_step_captures (phase 3),
            # not here, so dispatch_ms is where transfer cost appears.
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for layer_idx in layers_to_capture:
                for hook_name in hook_names:
                    manager.on_hook(layer_idx, hook_name, hidden)
            end_ev.record()
            torch.cuda.synchronize()

            # ── Phase 3: dispatch_step_captures (CPU fan-out + GPU→CPU copy)
            t2 = time.perf_counter()
            manager.dispatch_step_captures(plan)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            # Finalize to reset state.
            for rid in req_ids:
                manager.finalize_request(rid)
            # Clear sink result tables.
            for sink in manager._consumers:
                sink.clear()

            if phase == "measure":
                build_samples.append((t1 - t0) * 1000.0)
                hook_samples.append(start_ev.elapsed_time(end_ev))
                dispatch_samples.append((t3 - t2) * 1000.0)

    return {
        "build": build_samples,
        "hook": hook_samples,
        "dispatch": dispatch_samples,
    }


# ──────────────────────────────────────────────────────────── main

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CaptureManager plan building and dispatch"
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--batch-sizes", default="1,8,32",
        help="Comma-separated list of batch sizes"
    )
    parser.add_argument(
        "--num-consumers", default="1,2,4,8",
        help="Comma-separated list of consumer counts"
    )
    parser.add_argument(
        "--position-types", default="last_prompt,all_prompt",
        help="Comma-separated position selectors"
    )
    parser.add_argument(
        "--layer-counts", default="1,6,12",
        help="Comma-separated list of layer counts to capture"
    )
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--hook-name",
        default="post_mlp",
        help="(legacy) Single hook name. Ignored if --hook-sets is set.",
    )
    parser.add_argument(
        "--hook-sets",
        default=None,
        help=(
            "Semicolon-separated list of comma-separated hook-name sets. "
            "Each set becomes one sweep run, so "
            "'post_mlp;post_mlp,post_attn;post_mlp,post_attn,pre_mlp,pre_attn' "
            "runs three passes with 1, 2, and 4 hooks respectively. "
            "Default behavior (unset) uses --hook-name as a single-hook set."
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for capture manager benchmark")
        sys.exit(1)

    device = torch.device("cuda")
    cfg = MODEL_CONFIGS.get(args.model)
    if cfg is None:
        print(f"Unknown model {args.model!r}, using opt-125m config")
        cfg = MODEL_CONFIGS["facebook/opt-125m"]
    hidden_size = cfg["hidden_size"]
    num_hidden_layers = cfg["num_hidden_layers"]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    consumer_counts = [int(x) for x in args.num_consumers.split(",")]
    position_types = args.position_types.split(",")
    layer_counts = [int(x) for x in args.layer_counts.split(",")]

    # Parse hook sets.  ``--hook-sets a,b;c,d`` → [["a","b"], ["c","d"]].
    # Without it, fall back to a single-hook set from ``--hook-name``.
    if args.hook_sets:
        hook_sets = [
            [name.strip() for name in s.split(",") if name.strip()]
            for s in args.hook_sets.split(";")
            if s.strip()
        ]
    else:
        hook_sets = [[args.hook_name]]

    # Clamp layer counts to model's actual depth.
    layer_counts = [min(lc, num_hidden_layers) for lc in layer_counts]
    layer_counts = sorted(set(layer_counts))

    total = (
        len(batch_sizes) * len(consumer_counts) * len(position_types)
        * len(layer_counts) * len(hook_sets)
    )
    print(f"Capture manager benchmark: {args.model}")
    print(f"  batch_sizes={batch_sizes}, consumers={consumer_counts}")
    print(f"  position_types={position_types}, layer_counts={layer_counts}")
    print(f"  hook_sets={hook_sets}")
    print(f"  warmup={args.warmup}, iters={args.iters}, total configs={total}")
    print()

    all_results = []

    for batch_size, num_consumers, position_type, num_layers, hook_set in (
        itertools.product(
            batch_sizes, consumer_counts, position_types, layer_counts, hook_sets,
        )
    ):
        layers_to_capture = list(range(num_layers))
        req_ids = [f"req_{i:04d}" for i in range(batch_size)]
        total_tokens = batch_size * args.prompt_len

        hidden = torch.randn(
            total_tokens, hidden_size,
            device=device, dtype=torch.float16
        )

        batch_view = _make_batch_view(batch_size, args.prompt_len, req_ids)

        manager = _make_manager(
            num_consumers=num_consumers,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            layers_to_capture=layers_to_capture,
            hook_names=hook_set,
            position_type=position_type,
            device=device,
        )

        label = (
            f"bs={batch_size} nc={num_consumers} "
            f"pos={position_type} layers={num_layers} "
            f"hooks={len(hook_set)}({','.join(hook_set)})"
        )
        print(f"  {label}", flush=True)

        try:
            samples = _run_one(
                manager=manager,
                batch_view=batch_view,
                req_ids=req_ids,
                hidden=hidden,
                hook_names=hook_set,
                layers_to_capture=layers_to_capture,
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            del manager
            gc.collect()
            torch.cuda.empty_cache()
            continue

        build = compute_stats(samples["build"])
        hook = compute_stats(samples["hook"])
        dispatch = compute_stats(samples["dispatch"])

        print(
            f"    build p50={build.p50_ms:.3f}ms  "
            f"hook p50={hook.p50_ms:.3f}ms  "
            f"dispatch p50={dispatch.p50_ms:.3f}ms"
        )

        entry = {
            "batch_size": batch_size,
            "num_consumers": num_consumers,
            "position_type": position_type,
            "num_layers": num_layers,
            "num_hooks": len(hook_set),
            "hook_set": hook_set,
            "build_ms": build.to_dict(),
            "hook_ms": hook.to_dict(),
            "dispatch_ms": dispatch.to_dict(),
        }
        all_results.append(entry)

        del manager
        gc.collect()
        torch.cuda.empty_cache()

    # Write one result file with all sweep points.
    params = {
        "model": args.model,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "hook_sets": hook_sets,
        "prompt_len": args.prompt_len,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    write_result(
        benchmark="capture.manager",
        parameters=params,
        results={"sweep": all_results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary table
    print(f"\n{'=' * 110}")
    print(f"  CaptureManager Benchmark: {args.model}")
    print(f"{'=' * 110}")
    print(
        f"{'batch':>6} {'cons':>5} {'pos':<14} {'layers':>6} {'hooks':>5} "
        f"{'build_p50':>10} {'hook_p50':>10} {'disp_p50':>10}"
    )
    print("-" * 110)
    for r in all_results:
        print(
            f"{r['batch_size']:>6} {r['num_consumers']:>5} "
            f"{r['position_type']:<14} {r['num_layers']:>6} "
            f"{r['num_hooks']:>5} "
            f"{r['build_ms']['p50_ms']:>10.3f} "
            f"{r['hook_ms']['p50_ms']:>10.3f} "
            f"{r['dispatch_ms']['p50_ms']:>10.3f}"
        )
    print(f"{'=' * 110}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
