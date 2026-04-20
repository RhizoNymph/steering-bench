#!/usr/bin/env python3
"""Capture delivery-latency benchmark.

Answers: "for an online consumer, how long between the capture system
having an activation and my plugin seeing it?"

Two modes:

  --mode microbench
      Drives ``CaptureManager`` directly with a ``TimestampingSink``.
      Measures the dispatch-added delivery delay only — i.e. the cost
      the capture system imposes on top of the forward pass.  The
      microbench deliberately excludes intervening forward-pass
      kernels (which dominate real-inference latency); results are
      labelled ``dispatch_us`` so there is no confusion.

  --mode e2e
      Runs real ``LLM.generate()`` with a ``TimestampingConsumer`` and
      sweeps ``location ∈ {worker, driver}``.  This is the number a
      reward-consumer author actually wants: end-to-end latency from
      request submission to their ``on_capture`` callback firing.  The
      driver path exercises the worker→driver IPC bridge.

Results are written under ``benchmark="capture.latency.microbench"`` or
``"capture.latency.e2e"``.
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

from steering_bench.capture_consumers.consumers import (
    TimestampingConsumer,
    TimestampingSink,
)
from steering_bench.capture_consumers.runner import (
    get_model_config,
    make_prompts,
)
from steering_bench.output import write_result
from steering_bench.timing import compute_stats


# ──────────────────────────────────────────────────────────── helpers


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = min(len(s) - 1, int(len(s) * p))
    return s[k]


def _latency_summary(samples_us: list[float]) -> dict:
    """Produce a lightweight summary dict for a list of μs samples."""
    if not samples_us:
        return {"n": 0, "p50_us": None, "p99_us": None, "p999_us": None,
                "max_us": None, "mean_us": None}
    stats = compute_stats(samples_us)  # compute_stats is agnostic to units
    return {
        "n": stats.n,
        "mean_us": stats.mean_ms,  # unit-agnostic: samples are μs, field name is legacy
        "p50_us": stats.p50_ms,
        "p99_us": stats.p99_ms,
        "p999_us": _pct(samples_us, 0.999),
        "max_us": max(samples_us),
    }


# ──────────────────────────────────────────────────────────── microbench


def _run_microbench_one(
    batch_size: int,
    num_consumers: int,
    num_layers: int,
    position_type: str,
    hidden_size: int,
    num_hidden_layers: int,
    hook_name: str,
    prompt_len: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict:
    from vllm.v1.capture.manager import CaptureManager
    from vllm.v1.capture.plan import CaptureBatchView
    from vllm.v1.capture.types import CaptureSpec

    layers_to_capture = list(range(num_layers))
    req_ids = [f"req_{i:04d}" for i in range(batch_size)]
    total_tokens = batch_size * prompt_len
    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.float16)

    batch_view = CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=[prompt_len] * batch_size,
        num_computed_tokens=[0] * batch_size,
        num_scheduled_tokens=[prompt_len] * batch_size,
        token_offsets=[i * prompt_len for i in range(batch_size)],
    )

    spec = CaptureSpec(hooks={hook_name: layers_to_capture}, positions=position_type)
    sinks = tuple(TimestampingSink() for _ in range(num_consumers))
    specs = tuple(spec for _ in range(num_consumers))
    manager = CaptureManager(
        consumers=sinks,
        consumer_specs=specs,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        model_dtype=torch.float16,
        device=device,
    )

    # Per-consumer latency samples across all measured iterations.
    per_consumer_us: list[list[float]] = [[] for _ in range(num_consumers)]
    chunks_per_iter = 0

    try:
        for phase in ("warmup", "measure"):
            n = warmup if phase == "warmup" else iters
            for _ in range(n):
                for rid in req_ids:
                    manager.register_request(
                        req_id=rid, client_specs=None,
                        num_prompt_tokens=prompt_len,
                    )

                plan = manager.build_step_plan(batch_view)
                for layer_idx in layers_to_capture:
                    manager.on_hook(layer_idx, hook_name, hidden)
                torch.cuda.synchronize()

                # Drain defensively — no-op in the steady state, but
                # guards against any stamps left behind by error paths
                # or future refactors.
                for sink in sinks:
                    sink.drain_timestamps()

                t_dispatch_start = time.perf_counter_ns()
                manager.dispatch_step_captures(plan)
                torch.cuda.synchronize()

                if phase == "measure":
                    for idx, sink in enumerate(sinks):
                        stamps = sink.drain_timestamps()
                        if idx == 0 and chunks_per_iter == 0:
                            chunks_per_iter = len(stamps)
                        for _key, _step, ts_ns in stamps:
                            per_consumer_us[idx].append((ts_ns - t_dispatch_start) / 1000.0)
                else:
                    for sink in sinks:
                        sink.drain_timestamps()

                for rid in req_ids:
                    manager.finalize_request(rid)
                for sink in sinks:
                    sink.clear()
    finally:
        del manager
        gc.collect()
        torch.cuda.empty_cache()

    # Report per-consumer stats plus an aggregated "all_consumers" view.
    all_samples = [s for consumer_list in per_consumer_us for s in consumer_list]
    return {
        "chunks_per_iter_per_consumer": chunks_per_iter,
        "per_consumer": [_latency_summary(s) for s in per_consumer_us],
        "all": _latency_summary(all_samples),
    }


def _main_microbench(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for capture latency microbench", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    cfg = get_model_config(args.model)
    hidden_size = cfg["hidden_size"]
    num_hidden_layers = cfg["num_layers"]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    consumer_counts = [int(x) for x in args.num_consumers.split(",")]
    layer_counts = [int(x) for x in args.layer_counts.split(",")]
    layer_counts = sorted({min(lc, num_hidden_layers) for lc in layer_counts})
    position_types = args.position_types.split(",")

    total = (
        len(batch_sizes) * len(consumer_counts) * len(layer_counts) * len(position_types)
    )
    print(f"Capture-latency microbench: {args.model}")
    print(f"  batch_sizes={batch_sizes}, consumers={consumer_counts}")
    print(f"  layer_counts={layer_counts}, position_types={position_types}")
    print(f"  warmup={args.warmup}, iters={args.iters}, total configs={total}")
    print()

    all_results = []
    for bs, nc, nl, pt in itertools.product(
        batch_sizes, consumer_counts, layer_counts, position_types,
    ):
        label = f"bs={bs} nc={nc} layers={nl} pos={pt}"
        print(f"  {label}", flush=True)
        try:
            result = _run_microbench_one(
                batch_size=bs,
                num_consumers=nc,
                num_layers=nl,
                position_type=pt,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                hook_name=args.hook_name,
                prompt_len=args.prompt_len,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            all_results.append({
                "batch_size": bs, "num_consumers": nc, "num_layers": nl,
                "position_type": pt, "error": str(exc),
            })
            continue

        all_result = result["all"]
        print(
            f"    chunks/iter/consumer={result['chunks_per_iter_per_consumer']}  "
            f"p50={all_result['p50_us']:.1f}μs  "
            f"p99={all_result['p99_us']:.1f}μs  "
            f"p99.9={all_result['p999_us']:.1f}μs"
        )

        all_results.append({
            "batch_size": bs,
            "num_consumers": nc,
            "num_layers": nl,
            "position_type": pt,
            **result,
        })

    params = {
        "mode": "microbench",
        "model": args.model,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "hook_name": args.hook_name,
        "prompt_len": args.prompt_len,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    write_result(
        benchmark="capture.latency.microbench",
        parameters=params,
        results={"sweep": all_results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary.
    print(f"\n{'=' * 100}")
    print(f"  Capture Latency Microbench: {args.model}")
    print(f"{'=' * 100}")
    print(
        f"{'batch':>5} {'cons':>4} {'layers':>6} {'pos':<14} "
        f"{'chunks':>7} {'p50_us':>8} {'p99_us':>8} {'p99.9_us':>9}"
    )
    print("-" * 100)
    for r in all_results:
        if "error" in r:
            print(f"{r['batch_size']:>5} {r['num_consumers']:>4} "
                  f"{r['num_layers']:>6} {r['position_type']:<14}  "
                  f"ERROR: {r['error']}")
            continue
        a = r["all"]
        print(
            f"{r['batch_size']:>5} {r['num_consumers']:>4} "
            f"{r['num_layers']:>6} {r['position_type']:<14} "
            f"{r['chunks_per_iter_per_consumer']:>7} "
            f"{a['p50_us']:>8.1f} {a['p99_us']:>8.1f} {a['p999_us']:>9.1f}"
        )
    print(f"{'=' * 100}")
    print(f"Results written to {args.output_dir}")


# ──────────────────────────────────────────────────────────── e2e


def _wait_for_count(consumer, expected: int, timeout_s: float = 30.0) -> bool:
    """Block until ``consumer.count() >= expected`` or timeout.

    Driver-side consumers receive chunks via IPC, so on_capture may fire
    *after* ``llm.generate()`` returns.  Any latency number drained before
    this catches up is measuring "zero chunks" rather than the real
    pipeline delay.  Returns True on success, False on timeout.
    """
    deadline = time.perf_counter() + timeout_s
    while consumer.count() < expected:
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0.0005)
    return True


def _run_e2e_one(
    model: str,
    location: str,
    batch_size: int,
    num_layers: int,
    position_type: str,
    prompt_len: int,
    output_len: int,
    hook_name: str,
    model_cfg: dict,
    warmup: int,
    iters: int,
) -> dict:
    from vllm import LLM, SamplingParams

    layers_to_capture = list(range(min(num_layers, model_cfg["num_layers"])))
    hooks_dict = {hook_name: layers_to_capture}
    consumer = TimestampingConsumer(
        hooks=hooks_dict,
        positions=position_type,
        location=location,  # type: ignore[arg-type]
    )
    # One finalize → one on_capture call per (request, layer, hook) key.
    # Count from the hooks dict so this remains correct if the caller
    # ever passes multiple hook types.
    expected_per_iter = batch_size * sum(len(v) for v in hooks_dict.values())

    prompts = make_prompts(batch_size, prompt_len, model=model)
    sp = SamplingParams(max_tokens=output_len, temperature=0.0)
    sp_list = [sp] * batch_size

    print(f"    [loc={location} bs={batch_size} layers={num_layers}] loading model...",
          flush=True)
    llm = LLM(
        model=model,
        capture_consumers=[consumer],
        gpu_memory_utilization=0.9,
        max_model_len=512,
    )

    # Warmup — drain fully so state doesn't leak into measured iters.
    # If the first warmup iter fails to deliver any chunks, give up
    # immediately — the consumer is misconfigured or the driver bridge
    # isn't firing, and running 10 more warmup iters won't help.
    for wi in range(warmup):
        consumer.clear()
        llm.generate(prompts, sp_list)
        if not _wait_for_count(consumer, expected_per_iter):
            if wi == 0 and consumer.count() == 0:
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                return {"error": (
                    "warmup #0 delivered 0 chunks after 30s wait — "
                    "consumer likely not firing (check location / hooks / positions)"
                )}
        consumer.clear()

    # Measure.
    per_iter_samples_ms: list[float] = []
    per_iter_chunk_counts: list[int] = []
    drain_timeouts = 0
    for i in range(iters):
        consumer.clear()
        t_start = time.perf_counter_ns()
        llm.generate(prompts, sp_list)
        if not _wait_for_count(consumer, expected_per_iter):
            drain_timeouts += 1
            # Two consecutive timeouts → bail; fixture is broken and
            # further iterations will just waste minutes apiece.
            if drain_timeouts >= 2 and i <= drain_timeouts:
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                return {"error": f"drain timeout on {drain_timeouts} consecutive iters"}
        stamps = consumer.drain_timestamps()
        per_iter_chunk_counts.append(len(stamps))
        for _key, ts_ns in stamps:
            per_iter_samples_ms.append((ts_ns - t_start) / 1e6)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    if not per_iter_samples_ms:
        return {
            "error": "no chunks delivered",
            "drain_timeouts": drain_timeouts,
        }

    stats = compute_stats(per_iter_samples_ms)
    return {
        "chunks_per_iter_mean": sum(per_iter_chunk_counts) / len(per_iter_chunk_counts),
        "expected_per_iter": expected_per_iter,
        "drain_timeouts": drain_timeouts,
        "n_samples": stats.n,
        "mean_ms": stats.mean_ms,
        "p50_ms": stats.p50_ms,
        "p99_ms": stats.p99_ms,
        "p999_ms": _pct(per_iter_samples_ms, 0.999),
        "max_ms": max(per_iter_samples_ms),
    }


def _main_e2e(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for capture latency e2e", file=sys.stderr)
        sys.exit(1)

    model_cfg = get_model_config(args.model)
    locations = args.locations.split(",")
    if "worker" in locations:
        print(
            "ERROR: location='worker' is not supported in E2E mode — vLLM "
            "rejects pre-constructed CaptureConsumer instances with "
            "location='worker'. Use --mode microbench for worker-side "
            "latency measurements.",
            file=sys.stderr,
        )
        sys.exit(1)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    layer_counts = [int(x) for x in args.layer_counts.split(",")]
    position_types = args.position_types.split(",")

    total = (
        len(locations) * len(batch_sizes) * len(layer_counts) * len(position_types)
    )
    print(f"Capture-latency E2E: {args.model}")
    print(f"  locations={locations}  batch_sizes={batch_sizes}")
    print(f"  layer_counts={layer_counts}  position_types={position_types}")
    print(f"  warmup={args.warmup}  iters={args.iters}  total configs={total}")
    print()

    all_results = []
    for loc, bs, nl, pt in itertools.product(
        locations, batch_sizes, layer_counts, position_types,
    ):
        try:
            result = _run_e2e_one(
                model=args.model,
                location=loc,
                batch_size=bs,
                num_layers=nl,
                position_type=pt,
                prompt_len=args.prompt_len,
                output_len=args.output_len,
                hook_name=args.hook_name,
                model_cfg=model_cfg,
                warmup=args.warmup,
                iters=args.iters,
            )
        except torch.cuda.OutOfMemoryError:
            result = {"error": "OOM"}
        except Exception as exc:
            result = {"error": str(exc)}

        if "error" in result:
            print(f"    [loc={loc} bs={bs} layers={nl} pos={pt}] ERROR: {result['error']}")
        else:
            print(
                f"    [loc={loc} bs={bs} layers={nl} pos={pt}] "
                f"p50={result['p50_ms']:.1f}ms  p99={result['p99_ms']:.1f}ms  "
                f"chunks/iter≈{result['chunks_per_iter_mean']:.0f}"
            )

        all_results.append({
            "location": loc,
            "batch_size": bs,
            "num_layers": nl,
            "position_type": pt,
            **result,
        })

    params = {
        "mode": "e2e",
        "model": args.model,
        "hook_name": args.hook_name,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    write_result(
        benchmark="capture.latency.e2e",
        parameters=params,
        results={"sweep": all_results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary.
    print(f"\n{'=' * 110}")
    print(f"  Capture Latency E2E: {args.model}")
    print(f"{'=' * 110}")
    print(
        f"{'loc':<8} {'batch':>5} {'layers':>6} {'pos':<14} "
        f"{'mean_ms':>8} {'p50_ms':>8} {'p99_ms':>8} {'p99.9_ms':>9} {'max_ms':>8}"
    )
    print("-" * 110)
    for r in all_results:
        if "error" in r:
            print(f"{r['location']:<8} {r['batch_size']:>5} "
                  f"{r['num_layers']:>6} {r['position_type']:<14}  "
                  f"ERROR: {r['error']}")
            continue
        print(
            f"{r['location']:<8} {r['batch_size']:>5} "
            f"{r['num_layers']:>6} {r['position_type']:<14} "
            f"{r['mean_ms']:>8.1f} {r['p50_ms']:>8.1f} "
            f"{r['p99_ms']:>8.1f} {r['p999_ms']:>9.1f} {r['max_ms']:>8.1f}"
        )
    print(f"{'=' * 110}")
    print(f"Results written to {args.output_dir}")


# ──────────────────────────────────────────────────────────── main


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark capture delivery latency (microbench or E2E)"
    )
    parser.add_argument("--mode", choices=["microbench", "e2e"], required=True)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--hook-name", default="post_mlp")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")

    # Microbench-specific.
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--num-consumers", default="1,2,4,8",
                        help="(microbench only) consumer counts")
    parser.add_argument("--layer-counts", default="1,17,34")
    parser.add_argument("--position-types", default="last_prompt,all_prompt")
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Warmup iterations (default: 10 for microbench, 2 for e2e)",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Measured iterations (default: 50 for microbench, 10 for e2e)",
    )

    # E2E-specific.
    parser.add_argument(
        "--locations",
        default="driver",
        help=(
            "(e2e only) consumer locations to sweep. vLLM rejects "
            "pre-constructed CaptureConsumer instances with location='worker' "
            "(worker-side consumers must be registered as plugin dicts), so "
            "the E2E benchmark runs driver-only. For worker-side latency, "
            "see --mode microbench."
        ),
    )
    parser.add_argument("--output-len", type=int, default=32,
                        help="(e2e only) tokens to generate per request")

    args = parser.parse_args()

    # Fill in mode-appropriate defaults.  E2E is much slower per iter
    # (full model forward pass) so fewer iters go further.
    if args.warmup is None:
        args.warmup = 10 if args.mode == "microbench" else 2
    if args.iters is None:
        args.iters = 50 if args.mode == "microbench" else 10

    if args.mode == "microbench":
        _main_microbench(args)
    else:
        _main_e2e(args)


if __name__ == "__main__":
    main()
