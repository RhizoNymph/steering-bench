#!/usr/bin/env python3
"""Plugin-work benchmark: "how much CPU can a plugin burn per chunk?"

Drives a ``SimulatedWorkSink`` (microbench) or ``SimulatedWorkConsumer``
(e2e) with a configurable per-chunk work cost.  The sweep shows how
dispatch_ms and end-to-end throughput degrade as plugin work grows,
letting consumer authors budget their work per chunk.

Three work modes are offered:

  busy  — spin-wait on ``perf_counter_ns``.  Models a synchronous
          CPU-bound plugin.  Accurate down to sub-μs costs.
  sleep — ``time.sleep(work_us / 1e6)``.  Models yielding work.
          Meaningless below ~100 μs due to scheduler resolution; the
          sweep skips those points.
  queue — enqueue + worker thread drains with sleep.  Models the
          realistic pattern used by consumers that don't block the
          dispatch thread.  Captures backpressure when the queue fills.

Results are written under ``benchmark="capture.plugin_work.microbench"``
or ``"capture.plugin_work.e2e"``.
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
    SimulatedWorkConsumer,
    SimulatedWorkSink,
)
from steering_bench.capture_consumers.runner import (
    get_model_config,
    make_prompts,
)
from steering_bench.output import write_result
from steering_bench.timing import compute_stats


# Points below 100 μs are dropped for sleep/queue modes (scheduler-limited).
_SLEEP_MIN_US = 100.0


# ──────────────────────────────────────────────────────────── microbench


def _run_microbench_one(
    work_us: float,
    mode: str,
    num_consumers: int,
    batch_size: int,
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
    sinks = tuple(
        SimulatedWorkSink(work_us=work_us, mode=mode)  # type: ignore[arg-type]
        for _ in range(num_consumers)
    )
    specs = tuple(spec for _ in range(num_consumers))
    manager = CaptureManager(
        consumers=sinks,
        consumer_specs=specs,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        model_dtype=torch.float16,
        device=device,
    )

    # "Pipeline" here = dispatch_step_captures + finalize_request for all
    # requests + cuda sync.  The finalize step is included because queue
    # mode drains the worker-thread backlog on submit_finalize, so leaving
    # it out would hide the full cost of queue-mode plugin work.  Busy
    # and sleep modes do their work synchronously in submit_chunk, so the
    # numbers are directly comparable across modes.
    pipeline_samples_ms: list[float] = []

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

                t0 = time.perf_counter()
                manager.dispatch_step_captures(plan)
                for rid in req_ids:
                    manager.finalize_request(rid)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                if phase == "measure":
                    pipeline_samples_ms.append((t1 - t0) * 1000.0)

                for sink in sinks:
                    sink.clear()
    finally:
        for sink in sinks:
            try:
                sink.shutdown()
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                pass
        del manager
        gc.collect()
        torch.cuda.empty_cache()

    stats = compute_stats(pipeline_samples_ms)
    return {
        "n": stats.n,
        "pipeline_p50_ms": stats.p50_ms,
        "pipeline_p99_ms": stats.p99_ms,
        "pipeline_mean_ms": stats.mean_ms,
    }


def _main_microbench(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for plugin-work microbench", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    cfg = get_model_config(args.model)
    hidden_size = cfg["hidden_size"]
    num_hidden_layers = cfg["num_layers"]

    work_points = [float(x) for x in args.work_us.split(",")]
    work_modes = args.work_modes.split(",")
    consumer_counts = [int(x) for x in args.num_consumers.split(",")]
    num_layers = min(int(args.num_layers), num_hidden_layers)

    # Build the sweep list, dropping sleep/queue points below threshold.
    sweep: list[tuple[float, str, int]] = []
    for mode in work_modes:
        for w in work_points:
            if mode in ("sleep", "queue") and 0 < w < _SLEEP_MIN_US:
                continue
            for nc in consumer_counts:
                sweep.append((w, mode, nc))

    print(f"Plugin-work microbench: {args.model}")
    print(f"  work_us={work_points}  modes={work_modes}  consumers={consumer_counts}")
    print(f"  batch_size={args.batch_size}  num_layers={num_layers}  "
          f"position={args.position_type}")
    print(f"  warmup={args.warmup}  iters={args.iters}  total configs={len(sweep)}")
    print()

    all_results = []
    for work_us, mode, nc in sweep:
        label = f"mode={mode} work_us={work_us:.0f} cons={nc}"
        print(f"  {label}", flush=True)
        try:
            result = _run_microbench_one(
                work_us=work_us,
                mode=mode,
                num_consumers=nc,
                batch_size=args.batch_size,
                num_layers=num_layers,
                position_type=args.position_type,
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
                "work_us": work_us, "mode": mode, "num_consumers": nc,
                "error": str(exc),
            })
            continue

        print(
            f"    pipeline_p50={result['pipeline_p50_ms']:.3f}ms  "
            f"p99={result['pipeline_p99_ms']:.3f}ms"
        )
        all_results.append({
            "work_us": work_us,
            "mode": mode,
            "num_consumers": nc,
            **result,
        })

    params = {
        "mode": "microbench",
        "model": args.model,
        "batch_size": args.batch_size,
        "num_layers": num_layers,
        "position_type": args.position_type,
        "hook_name": args.hook_name,
        "prompt_len": args.prompt_len,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    write_result(
        benchmark="capture.plugin_work.microbench",
        parameters=params,
        results={"sweep": all_results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary.
    print(f"\n{'=' * 80}")
    print(f"  Plugin-Work Microbench: {args.model}")
    print(f"{'=' * 80}")
    print(f"{'mode':<8} {'work_us':>8} {'cons':>5} "
          f"{'pipe_p50':>10} {'pipe_p99':>10} {'pipe_mean':>10}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['mode']:<8} {r['work_us']:>8.0f} "
                  f"{r['num_consumers']:>5}  ERROR: {r['error']}")
            continue
        print(
            f"{r['mode']:<8} {r['work_us']:>8.0f} {r['num_consumers']:>5} "
            f"{r['pipeline_p50_ms']:>10.3f} {r['pipeline_p99_ms']:>10.3f} "
            f"{r['pipeline_mean_ms']:>10.3f}"
        )
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


# ──────────────────────────────────────────────────────────── e2e


def _wait_for_count(consumer, expected: int, timeout_s: float = 60.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while consumer.count() < expected:
        if time.perf_counter() >= deadline:
            return False
        time.sleep(0.0005)
    return True


def _run_e2e_one(
    model: str,
    work_us: float,
    mode: str,
    batch_size: int,
    num_layers: int,
    position_type: str,
    prompt_len: int,
    output_len: int,
    hook_name: str,
    model_cfg: dict,
    location: str,
    warmup: int,
    iters: int,
) -> dict:
    from vllm import LLM, SamplingParams

    layers_to_capture = list(range(min(num_layers, model_cfg["num_layers"])))
    capture_consumers: list | None
    consumer = None
    expected_per_iter = 0
    if work_us < 0:
        # Sentinel for "baseline, no consumer at all".
        capture_consumers = None
    else:
        hooks_dict = {hook_name: layers_to_capture}
        consumer = SimulatedWorkConsumer(
            hooks=hooks_dict,
            positions=position_type,
            work_us=work_us,
            mode=mode,  # type: ignore[arg-type]
            location=location,  # type: ignore[arg-type]
        )
        capture_consumers = [consumer]
        # One on_capture call per (request, layer, hook) key.
        expected_per_iter = batch_size * sum(len(v) for v in hooks_dict.values())

    prompts = make_prompts(batch_size, prompt_len, model=model)
    sp = SamplingParams(max_tokens=output_len, temperature=0.0)
    sp_list = [sp] * batch_size

    label_mode = "baseline" if work_us < 0 else f"{mode}@{work_us:.0f}us"
    print(f"    [{label_mode} bs={batch_size}] loading model...", flush=True)
    llm = LLM(
        model=model,
        capture_consumers=capture_consumers,
        gpu_memory_utilization=0.9,
        max_model_len=512,
    )

    try:
        # Warmup — drain before next iteration to avoid bleed-over.
        for wi in range(warmup):
            llm.generate(prompts, sp_list)
            if consumer is not None:
                drained = _wait_for_count(consumer, expected_per_iter)
                if not drained and wi == 0 and consumer.count() == 0:
                    return {"error": "warmup #0 delivered 0 chunks (consumer not firing)"}
                consumer.clear()

        # Measure.  Wall clock includes drain wait so any plugin work that
        # spills past forward-pass time becomes visible as extra ms.  If
        # the plugin is fast enough to fully overlap with forward work,
        # the wait is near-zero and overhead ≈ 0.
        samples_ms: list[float] = []
        drain_timeouts = 0
        for i in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sp_list)
            if consumer is not None:
                if not _wait_for_count(consumer, expected_per_iter):
                    drain_timeouts += 1
                    # Bail after 2 consecutive timeouts — subsequent iters
                    # will just burn the timeout budget again.
                    if drain_timeouts >= 2 and i <= drain_timeouts:
                        return {"error": f"drain timeout on {drain_timeouts} iters"}
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
            if consumer is not None:
                consumer.clear()
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    if not samples_ms:
        return {"error": "no samples collected"}
    stats = compute_stats(samples_ms)
    tokens_per_sec = (batch_size * output_len) / (stats.mean_ms / 1000.0)
    return {
        "n": stats.n,
        "mean_ms": stats.mean_ms,
        "p50_ms": stats.p50_ms,
        "p99_ms": stats.p99_ms,
        "tokens_per_sec": tokens_per_sec,
        "expected_per_iter": expected_per_iter,
        "drain_timeouts": drain_timeouts,
    }


def _main_e2e(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for plugin-work e2e", file=sys.stderr)
        sys.exit(1)

    model_cfg = get_model_config(args.model)
    work_points = [float(x) for x in args.work_us.split(",")]
    work_modes = args.work_modes.split(",")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    num_layers = min(int(args.num_layers), model_cfg["num_layers"])

    # Sweep: first a baseline per batch (no consumer), then (mode, work_us) points.
    print(f"Plugin-work E2E: {args.model}")
    print(f"  work_us={work_points}  modes={work_modes}  batch_sizes={batch_sizes}")
    print(f"  num_layers={num_layers}  location={args.location}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print()

    all_results = []
    baseline_tps: dict[int, float] = {}

    for bs in batch_sizes:
        # Baseline (no consumer).  Use work_us=-1 as the sentinel.
        print(f"  [baseline bs={bs}]", flush=True)
        try:
            result = _run_e2e_one(
                model=args.model,
                work_us=-1.0,
                mode="busy",
                batch_size=bs,
                num_layers=num_layers,
                position_type=args.position_type,
                prompt_len=args.prompt_len,
                output_len=args.output_len,
                hook_name=args.hook_name,
                model_cfg=model_cfg,
                location=args.location,
                warmup=args.warmup,
                iters=args.iters,
            )
            baseline_tps[bs] = result["tokens_per_sec"]
            print(f"    tps={result['tokens_per_sec']:.0f}  "
                  f"mean={result['mean_ms']:.1f}ms")
            all_results.append({
                "work_us": 0.0, "mode": "baseline", "batch_size": bs,
                "overhead_pct": 0.0, **result,
            })
        except Exception as exc:
            print(f"    ERROR: {exc}")
            all_results.append({
                "work_us": 0.0, "mode": "baseline", "batch_size": bs,
                "error": str(exc),
            })
            continue

        for mode, work_us in itertools.product(work_modes, work_points):
            if mode in ("sleep", "queue") and 0 < work_us < _SLEEP_MIN_US:
                continue
            print(f"  [{mode}@{work_us:.0f}us bs={bs}]", flush=True)
            try:
                result = _run_e2e_one(
                    model=args.model,
                    work_us=work_us,
                    mode=mode,
                    batch_size=bs,
                    num_layers=num_layers,
                    position_type=args.position_type,
                    prompt_len=args.prompt_len,
                    output_len=args.output_len,
                    hook_name=args.hook_name,
                    model_cfg=model_cfg,
                    location=args.location,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                all_results.append({
                    "work_us": work_us, "mode": mode, "batch_size": bs,
                    "error": str(exc),
                })
                continue

            overhead = None
            base = baseline_tps.get(bs)
            if base is not None and base > 0:
                overhead = (base - result["tokens_per_sec"]) / base * 100.0
            overhead_str = (
                f"  overhead={overhead:+.1f}%" if overhead is not None else ""
            )
            print(
                f"    tps={result['tokens_per_sec']:.0f}  "
                f"mean={result['mean_ms']:.1f}ms{overhead_str}"
            )
            all_results.append({
                "work_us": work_us, "mode": mode, "batch_size": bs,
                "overhead_pct": overhead, **result,
            })

    # Compute work budget per (mode, batch_size): largest work_us with overhead < threshold.
    budget_pct = args.budget_threshold_pct
    budgets = []
    for mode in work_modes:
        for bs in batch_sizes:
            candidates = [
                r for r in all_results
                if "error" not in r and r.get("mode") == mode
                and r.get("batch_size") == bs
                and r.get("overhead_pct") is not None
                and r["overhead_pct"] < budget_pct
            ]
            if not candidates:
                budget_us = 0.0
            else:
                budget_us = max(r["work_us"] for r in candidates)
            budgets.append({
                "mode": mode,
                "batch_size": bs,
                "threshold_pct": budget_pct,
                "budget_us": budget_us,
            })

    params = {
        "mode": "e2e",
        "model": args.model,
        "num_layers": num_layers,
        "position_type": args.position_type,
        "hook_name": args.hook_name,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "location": args.location,
        "warmup": args.warmup,
        "iters": args.iters,
        "budget_threshold_pct": budget_pct,
    }
    write_result(
        benchmark="capture.plugin_work.e2e",
        parameters=params,
        results={"sweep": all_results, "budgets": budgets},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary.
    print(f"\n{'=' * 90}")
    print(f"  Plugin-Work E2E: {args.model}")
    print(f"{'=' * 90}")
    print(f"{'mode':<10} {'batch':>5} {'work_us':>8} "
          f"{'mean_ms':>10} {'tps':>8} {'overhead':>10}")
    print("-" * 90)
    for r in all_results:
        if "error" in r:
            print(
                f"{r.get('mode', '?'):<10} "
                f"{r.get('batch_size', 0):>5} "
                f"{r.get('work_us', 0):>8.0f}  "
                f"ERROR: {r['error']}"
            )
            continue
        ov = r.get("overhead_pct")
        ov_str = f"{ov:+.1f}%" if ov is not None else "baseline"
        print(
            f"{r['mode']:<10} {r['batch_size']:>5} {r['work_us']:>8.0f} "
            f"{r['mean_ms']:>10.1f} {r['tokens_per_sec']:>8.0f} {ov_str:>10}"
        )
    print(f"\nPlugin work budgets (overhead < {budget_pct}%):")
    for b in budgets:
        print(f"  mode={b['mode']:<8} batch={b['batch_size']:>3}  "
              f"budget = {b['budget_us']:.0f} μs per chunk")
    print(f"{'=' * 90}")
    print(f"Results written to {args.output_dir}")


# ──────────────────────────────────────────────────────────── main


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark capture plugin work budget (microbench or E2E)"
    )
    parser.add_argument("--mode", choices=["microbench", "e2e"], required=True)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--hook-name", default="post_mlp")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--position-type", default="last_prompt")
    parser.add_argument("--num-layers", default=17,
                        help="Layer count to capture (clamped to model depth)")
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")

    # Work-sweep common.  E2E mode needs higher values because plugin work
    # overlaps with forward-pass time — saturation on gemma-3-4b requires
    # tens of milliseconds per on_capture call at batch=8.
    parser.add_argument(
        "--work-us", default="0,100,1000,10000,50000,200000",
        help="Comma-separated work_us points to sweep",
    )
    parser.add_argument("--work-modes", default="busy,sleep,queue",
                        help="Work modes to sweep (busy/sleep/queue)")

    # Microbench-specific.
    parser.add_argument("--num-consumers", default="1,2,4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Warmup iterations (default: 10 for microbench, 2 for e2e)",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Measured iterations (default: 50 for microbench, 5 for e2e)",
    )

    # E2E-specific.
    parser.add_argument("--batch-sizes", default="8",
                        help="(e2e only) batch sizes to sweep")
    parser.add_argument("--output-len", type=int, default=32,
                        help="(e2e only) tokens to generate per request")
    parser.add_argument("--location", default="driver",
                        help="(e2e only) consumer location (worker/driver)")
    parser.add_argument("--budget-threshold-pct", type=float, default=5.0,
                        help=("(e2e only) overhead threshold (%%) used to "
                              "compute the plugin work budget"))

    args = parser.parse_args()

    # Mode-appropriate iter defaults.  E2E runs a full model forward
    # pass per iter plus drain-wait (up to ~30s at work_us=200ms), so
    # iters=5 is already 5-20 minutes per config point.
    if args.warmup is None:
        args.warmup = 10 if args.mode == "microbench" else 2
    if args.iters is None:
        args.iters = 50 if args.mode == "microbench" else 5

    if args.mode == "microbench":
        _main_microbench(args)
    else:
        _main_e2e(args)


if __name__ == "__main__":
    main()
