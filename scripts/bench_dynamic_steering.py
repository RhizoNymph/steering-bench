#!/usr/bin/env python3
"""Dynamic-steering + capture-consumer end-to-end overhead benchmark.

Measures the wall-clock ``generate()`` throughput overhead of the dynamic
steering / capture-consumer tiers against a no-capture, no-steering baseline,
on a real gemma4 model (the only architecture carrying the capture taps that
every consumer — and thus all dynamic steering — reads from).

Six arms (see ``src/steering_bench/capture_consumers/bench_consumers.py``):

  off            no capture, no steering (cold path, reference)
  capture_async  on_capture reader, no steer (preexisting async capture)
  capture_sync   on_step reader, no steer (new sync path)
  steer_async    steer via the action queue (1-3 step latency)
  steer_sync     steer via on_step (exactly-one-step latency)
  steer_dynamic  in-graph monitor gating the tier (same-token)

Decomposition the arms give:
  off -> capture_async      capture gather/dispatch pipeline cost
  capture_async -> capture_sync   per-step on_step vs finalize on_capture
  capture_sync -> steer_sync      the steering kernel itself (steady-state)
  steer_async / sync / dynamic    the three tier transports head-to-head

Each (arm, batch_size) cell runs in a FRESH subprocess: the dynamic-steering
action queue is process-global, and vLLM's residual weight memory does not free
cleanly across LLM instances — a subprocess per cell keeps state and memory
clean. The parent re-invokes this script with ``--cell`` for each cell.

Requires CUDA + a gemma4 model. Example:

    VLLM_USE_FLASHINFER_SAMPLER=0 \
    uv run scripts/bench_dynamic_steering.py \
      --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf --layer 30 \
      --batch-sizes 1,8,32 --output-len 64 --prompt-len 64
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# In-process engine core: the default forks a worker subprocess after the
# parent touches CUDA, and PyTorch refuses to re-init CUDA in a forked child.
# Running the engine in-process also keeps the timing free of IPC. Mirrors
# bench_steering_with_capture.py / nsys_target.py.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# node2/node0 CUDA/CUB toolchain fails the flashinfer sampling-kernel JIT;
# unrelated to steering. Harmless elsewhere.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# (name, enable_steering, consumer-spec builder taking the probe/steer layer)
ARMS: dict[str, tuple[bool, Any]] = {
    "off": (False, lambda layer, norm: None),
    "capture_async": (
        False,
        lambda layer, norm: [
            {"name": "bench_capture_async", "params": {"layer": layer}}
        ],
    ),
    "capture_sync": (
        False,
        lambda layer, norm: [
            {"name": "bench_capture_sync", "params": {"layer": layer}}
        ],
    ),
    "steer_async": (
        True,
        lambda layer, norm: [
            {"name": "bench_steer_async", "params": {"layer": layer, "norm": norm}}
        ],
    ),
    "steer_sync": (
        True,
        lambda layer, norm: [
            {"name": "bench_steer_sync", "params": {"layer": layer, "norm": norm}}
        ],
    ),
    "steer_dynamic": (
        True,
        lambda layer, norm: [
            {"name": "bench_steer_dynamic", "params": {"layer": layer, "norm": norm}}
        ],
    ),
    "steer_per_request": (
        True,
        lambda layer, norm: [
            {"name": "bench_steer_per_request",
             "params": {"layer": layer, "norm": norm}}
        ],
    ),
}

ARM_ORDER = list(ARMS.keys())


# --------------------------------------------------------------------------
# Cell mode — build one LLM, measure, emit a JSON line
# --------------------------------------------------------------------------


def _make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    return [" ".join(["hello"] * words_needed)] * num_prompts


def _gpu_sample() -> dict[str, int]:
    """Current SM clock (MHz) and temperature (C) — for thermal-drift
    transparency. Long bs cells heat the GPU; if clocks throttle across a
    sweep, cross-cell deltas reflect thermal headroom, not code. Lock clocks
    (``nvidia-smi -lgc``) for clean numbers, or rescale post-hoc with
    ``rescale_clocks.py`` using the recorded clock."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.current.graphics,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        clk, temp = (x.strip() for x in out.split(","))
        return {"gpu_clock_mhz": int(clk), "gpu_temp_c": int(temp)}
    except Exception:
        return {}


def _run_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    enable_steering, build = ARMS[args.arm]
    consumers = build(args.layer, args.norm)

    prompts = _make_prompts(args.batch_size, args.prompt_len)
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.0, seed=0)
    sp_list = [sp] * args.batch_size

    kwargs: dict[str, Any] = dict(
        model=args.model,
        capture_consumers=consumers,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.prompt_len + args.output_len + 64,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    if enable_steering:
        kwargs["enable_steering"] = True
        # Per-request arm needs one override-pool row per concurrent request;
        # tier arms use a single global config, so 4 is plenty for them.
        if args.arm == "steer_per_request":
            kwargs["max_dynamic_steering_configs"] = max(8, args.batch_size + 8)
        else:
            kwargs["max_dynamic_steering_configs"] = 4

    llm = LLM(**kwargs)
    try:
        for _ in range(args.warmup):
            llm.generate(prompts, sp_list)
        gpu_start = _gpu_sample()
        samples_ms: list[float] = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sp_list)
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        gpu_end = _gpu_sample()
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    from steering_bench.timing import compute_stats

    stats = compute_stats(samples_ms)
    tps = (args.batch_size * args.output_len) / (stats.mean_ms / 1000.0)
    return {
        "arm": args.arm,
        "batch_size": args.batch_size,
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p90_ms": stats.p90_ms,
        "stddev_ms": stats.stddev_ms,
        "tokens_per_sec": tps,
        # Thermal-drift transparency: clock/temp at the start vs end of the
        # measured iters. A clock drop across a sweep means throttling is
        # confounding cross-cell deltas — lock clocks or rescale.
        "gpu_clock_start_mhz": gpu_start.get("gpu_clock_mhz"),
        "gpu_clock_end_mhz": gpu_end.get("gpu_clock_mhz"),
        "gpu_temp_end_c": gpu_end.get("gpu_temp_c"),
    }


# --------------------------------------------------------------------------
# Parent mode — fan cells out into subprocesses, aggregate, report
# --------------------------------------------------------------------------


def _cell_subprocess(args: argparse.Namespace, arm: str, batch_size: int) -> dict[str, Any]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--arm", arm, "--batch-size", str(batch_size),
        "--model", args.model, "--layer", str(args.layer),
        "--norm", str(args.norm), "--prompt-len", str(args.prompt_len),
        "--output-len", str(args.output_len), "--warmup", str(args.warmup),
        "--iters", str(args.iters), "--gpu-mem-util", str(args.gpu_mem_util),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return {"arm": arm, "batch_size": batch_size, "error": f"exit {proc.returncode}"}
    for line in proc.stdout.splitlines():
        if line.startswith("CELL_RESULT "):
            return json.loads(line[len("CELL_RESULT "):])
    return {"arm": arm, "batch_size": batch_size, "error": "no CELL_RESULT emitted"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "Models/gemma-4-31B-it-Q4_K_S.gguf"))
    parser.add_argument("--layer", type=int, default=30, help="probe/steer layer")
    parser.add_argument("--norm", type=float, default=8.0, help="steer vector L2 norm")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--gpu-mem-util", type=float, default=0.92)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable cudagraphs (default: cudagraphs on)")
    parser.add_argument("--arms", default="", help="comma-subset of arms (default: all)")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="seconds to idle before each cell so the GPU "
                             "returns to a comparable thermal state (long bs "
                             "cells otherwise heat the GPU and throttle later "
                             "cells; locking clocks with `nvidia-smi -lgc` is "
                             "the cleaner fix)")
    parser.add_argument("--output-dir", default="results/dynamic_steering/")
    parser.add_argument("--tag", default="")
    # Internal single-cell mode.
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:
        result = _run_cell(args)
        print("CELL_RESULT " + json.dumps(result), flush=True)
        return

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    from steering_bench.output import write_result

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    arms = [a for a in ARM_ORDER if a in set(args.arms.split(","))] if args.arms else ARM_ORDER

    print(f"Dynamic-steering benchmark: {args.model}")
    print(f"  arms={arms}")
    print(f"  batch_sizes={batch_sizes} prompt_len={args.prompt_len} "
          f"output_len={args.output_len} layer={args.layer} "
          f"enforce_eager={args.enforce_eager}")
    print(f"  warmup={args.warmup} iters={args.iters}\n")

    all_results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        print(f"--- batch_size={batch_size} ---")
        baseline_mean: float | None = None
        for arm in arms:
            if args.cooldown > 0:
                time.sleep(args.cooldown)
            r = _cell_subprocess(args, arm, batch_size)
            if "error" in r:
                print(f"    [{arm}] ERROR: {r['error']}")
                all_results.append(r)
                continue
            if arm == "off":
                baseline_mean = r["mean_ms"]
            overhead = (
                (r["mean_ms"] - baseline_mean) / baseline_mean * 100.0
                if baseline_mean else None
            )
            r["overhead_pct"] = overhead
            ov = f"{overhead:+.1f}%" if overhead is not None else "baseline"
            clk = r.get("gpu_clock_end_mhz")
            clk_s = f" clk={clk}MHz" if clk else ""
            print(f"    [{arm:<14}] mean={r['mean_ms']:.1f}ms "
                  f"tps={r['tokens_per_sec']:.0f} overhead={ov}{clk_s}")
            write_result(
                benchmark="steering.dynamic",
                parameters={
                    "model": args.model, "arm": arm, "batch_size": batch_size,
                    "prompt_len": args.prompt_len, "output_len": args.output_len,
                    "layer": args.layer, "norm": args.norm,
                    "enforce_eager": args.enforce_eager,
                    "warmup": args.warmup, "iters": args.iters,
                },
                results={k: v for k, v in r.items() if k not in ("arm",)},
                output_dir=args.output_dir,
                tag=args.tag,
            )
            all_results.append(r)
        print()

    # Summary
    print("=" * 78)
    print(f"  Dynamic-steering benchmark: {Path(args.model).name}")
    print("=" * 78)
    print(f"{'batch':>6} {'arm':<14} {'mean_ms':>10} {'p90_ms':>9} "
          f"{'tps':>9} {'overhead':>10}")
    print("-" * 78)
    for r in all_results:
        if "error" in r:
            print(f"{r['batch_size']:>6} {r['arm']:<14} {'ERROR':>10}")
            continue
        ov = r.get("overhead_pct")
        ov_s = f"{ov:+.1f}%" if ov is not None else "baseline"
        print(f"{r['batch_size']:>6} {r['arm']:<14} {r['mean_ms']:>10.1f} "
              f"{r['p90_ms']:>9.1f} {r['tokens_per_sec']:>9.0f} {ov_s:>10}")
    print("=" * 78)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
