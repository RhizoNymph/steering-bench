#!/usr/bin/env python3
"""Dynamic-steering + capture-consumer end-to-end overhead benchmark.

Measures the wall-clock ``generate()`` throughput overhead of the dynamic
steering / capture-consumer tiers against a no-capture, no-steering baseline,
on a real model carrying the capture taps that every consumer — and thus all
dynamic steering — reads from.

Arms (see ``src/steering_bench/capture_consumers/bench_consumers.py``):

  off            no capture, no steering (cold path, reference)
  capture_async  on_capture reader, no steer (preexisting async capture)
  capture_sync   on_step reader, no steer (new sync path)
  steer_async    steer via the action queue (1-3 step latency), global tier
  steer_sync     steer via on_step (exactly-one-step latency), global tier
  steer_dynamic  in-graph GLOBAL monitor gating the tier (same-token)
  steer_override per-request dynamic-override pool (one row per live request)
  steer_rowmon   per-request override + PER-ROW in-graph monitor (needs the
                 ``enable_row_monitor`` engine opt-in)

Decomposition the arms give:
  off -> capture_async      capture gather/dispatch pipeline cost
  capture_async -> capture_sync   per-step on_step vs finalize on_capture
  capture_sync -> steer_sync      the steering kernel itself (steady-state)
  steer_async / sync / dynamic    the three tier transports head-to-head
  steer_override -> steer_rowmon  the per-row in-graph monitor cost

Each (arm, batch_size) cell runs in a FRESH subprocess: the dynamic-steering
action queue is process-global, and vLLM's residual weight memory does not free
cleanly across LLM instances — a subprocess per cell keeps state and memory
clean. The parent re-invokes this script with ``--cell`` for each cell.

After warmup, every cell asserts the arm is genuinely ACTIVE (worker-side
status RPC for steering arms + the in-process consumer registry for capture
arms) and fails the cell loudly otherwise — the old harness silently measured
an inactive steering arm at least once (nothing-active short-circuit / a stale
hook name). See ``_assert_arm_active``.

Requires CUDA + a model with capture taps. Example (gemma-3-4b-it, 34 layers):

    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    uv run scripts/bench_dynamic_steering.py \
      --model google/gemma-3-4b-it --num-model-layers 34 \
      --steer-layers 17 --steer-hooks post_block \
      --batch-sizes 1,8,16,32 --output-len 64 --prompt-len 64
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
# Running the engine in-process also keeps the timing free of IPC AND — with
# the worker in this process — lets the activation assertion read consumer
# state directly. Mirrors bench_steering_with_capture.py / nsys_target.py.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# node2/node0 CUDA/CUB toolchain fails the flashinfer sampling-kernel JIT;
# unrelated to steering. Harmless elsewhere.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Arm -> (enable_steering, enable_row_monitor). The consumer entry-point name
# is always ``"bench_" + arm``.
ARM_FLAGS: dict[str, tuple[bool, bool]] = {
    "off": (False, False),
    "capture_async": (False, False),
    "capture_sync": (False, False),
    "steer_async": (True, False),
    "steer_sync": (True, False),
    "steer_dynamic": (True, False),
    "steer_override": (True, False),
    "steer_rowmon": (True, True),
    # Distinct-vector-per-request override (merged from the parallel bench
    # effort); measured like the override arm via applied-counter activation.
    "steer_per_request": (True, False),
}
ARM_ORDER = list(ARM_FLAGS.keys())
STEER_ARMS = {a for a, (s, _) in ARM_FLAGS.items() if s}
CAPTURE_ARMS = {"capture_async", "capture_sync"}
# Per-request override arms need one dynamic-pool row per concurrent request.
PERREQ_ARMS = {"steer_override", "steer_rowmon", "steer_per_request"}


# --------------------------------------------------------------------------
# Site resolution: ``spread:N`` -> N evenly spaced layers across the model
# --------------------------------------------------------------------------


def resolve_layers(spec: str, num_layers: int) -> list[int]:
    """Resolve a ``--steer-layers`` spec to concrete layer indices.

    ``spread:N`` -> N evenly spaced layers across ``num_layers``; a comma list
    of ints is returned as-is. ``spread:1`` is the middle layer.
    """
    spec = spec.strip()
    if spec.startswith("spread:"):
        n = int(spec.split(":", 1)[1])
        if n <= 0:
            raise ValueError(f"spread:N needs N>=1, got {spec!r}")
        if n == 1:
            return [int((num_layers - 1) / 2 + 0.5)]
        if n >= num_layers:
            return list(range(num_layers))
        import numpy as np

        pts = np.linspace(0, num_layers - 1, n)
        return sorted({int(x + 0.5) for x in pts})
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def _consumer_params(args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    hooks = [h.strip() for h in args.steer_hooks.split(",") if h.strip()]
    mon_layer = args.monitor_layer if args.monitor_layer >= 0 else layers[0]
    mon_hook = args.monitor_hook or hooks[0]
    return {
        "layers": layers,
        "hooks": hooks,
        "norm": args.norm,
        "monitor_layer": mon_layer,
        "monitor_hook": mon_hook,
    }


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


def _dynamic_status(llm: Any) -> list[dict[str, Any]]:
    """Per-worker ``get_dynamic_steering_status`` payloads (one per worker;
    tp=pp=1 -> a single element)."""
    res = llm.collective_rpc("get_dynamic_steering_status")
    return list(res) if isinstance(res, (list, tuple)) else [res]


class _ArmInactive(RuntimeError):
    """Raised when a cell's arm is not genuinely active after warmup."""


def _assert_arm_active(arm: str, llm: Any, batch_size: int) -> dict[str, Any]:
    """Fail loudly unless the arm is genuinely active after warmup.

    Robust, post-warmup, OUT of the timed region. For steering arms we read the
    worker ``get_dynamic_steering_status`` RPC:

      * tier arms (steer_async/sync/dynamic) send a decode-phase
        ``SteeringVectorUpdate`` -> the §5.4 DYNAMIC TIER (``dynamic_tier``);
        ``active`` is True only if the update was APPLIED (a rejected update —
        e.g. a stale hook — leaves the tier empty), so this alone proves the
        arm is live;
      * steer_dynamic additionally installs the GLOBAL monitor -> assert
        ``monitor.active``;
      * per-request arms (steer_override/rowmon) route each request to its own
        dynamic row. The pool ``in_use`` drops as requests finish, so we use
        the PERSISTENT per-source ``applied`` counter instead (>= batch_size,
        or >= 2*batch_size for rowmon which emits override+monitor per req);
      * steer_rowmon requires ``row_monitor.enabled`` (the engine opt-in — the
        offline ``LLM(...)`` path has historically dropped such bools);
      * EVERY steering arm must have ``rejected == 0`` — a nonzero reject means
        a misconfigured hook/target, i.e. the arm is (partly) inactive.

    Capture arms have no worker status RPC; with multiprocessing=0 the worker
    is in-process, so we read the consumer registry's step counters directly.

    Returns a small diagnostics dict recorded in the result.
    """
    if arm == "off":
        return {"active": True, "check": "baseline"}

    if arm in CAPTURE_ARMS:
        from steering_bench.capture_consumers.bench_consumers import (
            iter_live_consumers,
        )

        steps = [c._steps for c in iter_live_consumers()]
        total = sum(steps)
        if total <= 0:
            raise _ArmInactive(
                f"{arm}: no consumer step recorded (steps={steps}); capture "
                "spec likely inactive (stale hook?)"
            )
        return {"active": True, "check": "consumer_steps", "consumer_steps": total}

    # Steering arms: worker status RPC.
    workers = _dynamic_status(llm)
    diag: dict[str, Any] = {"check": "status_rpc", "workers": len(workers)}
    for wi, st in enumerate(workers):
        if not st.get("steering_initialized"):
            raise _ArmInactive(f"{arm}: worker {wi} steering not initialized")
        # No rejected updates anywhere — a reject means misconfiguration.
        apply_stats = st.get("apply_stats") or {}
        for source, counts in apply_stats.items():
            if counts.get("rejected", 0) != 0:
                raise _ArmInactive(
                    f"{arm}: worker {wi} source {source!r} rejected="
                    f"{counts['rejected']} (misconfigured update)"
                )
        aq = st.get("action_queue") or {}
        if aq.get("rejected", 0):
            raise _ArmInactive(
                f"{arm}: worker {wi} action_queue rejected={aq['rejected']}"
            )

        if arm in ("steer_async", "steer_sync", "steer_dynamic"):
            tier = st.get("dynamic_tier") or {}
            if not tier.get("active"):
                raise _ArmInactive(
                    f"{arm}: worker {wi} dynamic tier not active "
                    f"(update not applied): {tier}"
                )
            if arm == "steer_dynamic":
                mon = st.get("monitor") or {}
                if not mon.get("active"):
                    raise _ArmInactive(
                        f"{arm}: worker {wi} global monitor not active: {mon}"
                    )
            diag.setdefault("tier_active", True)

        elif arm in PERREQ_ARMS:
            source = f"bench_{arm}"
            applied = (apply_stats.get(source) or {}).get("applied", 0)
            need = batch_size * (2 if arm == "steer_rowmon" else 1)
            if applied < need:
                raise _ArmInactive(
                    f"{arm}: worker {wi} applied={applied} < {need} "
                    f"(source {source!r}); override pool not driven"
                )
            if arm == "steer_rowmon":
                rm = st.get("row_monitor") or {}
                if not rm.get("enabled"):
                    raise _ArmInactive(
                        f"{arm}: worker {wi} row_monitor not enabled — the "
                        "enable_row_monitor engine flag was dropped"
                    )
                diag.setdefault("row_monitor_enabled", True)
            diag.setdefault("applied", applied)

    diag["active"] = True
    return diag


def _run_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    enable_steering, enable_row_monitor = ARM_FLAGS[args.arm]
    layers = resolve_layers(args.steer_layers, args.num_model_layers)

    consumers: list[dict[str, Any]] | None = None
    if args.arm != "off":
        consumers = [
            {"name": f"bench_{args.arm}", "params": _consumer_params(args, layers)}
        ]

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
        # Per-request arms need one dynamic-pool row per concurrent request;
        # +4 headroom avoids an exactly-full pool at the batch/generate seam.
        # Tier arms use a single global config, so 4 is plenty for them.
        kwargs["max_dynamic_steering_configs"] = (
            max(args.batch_size + 4, 8) if args.arm in PERREQ_ARMS else 4
        )
        if enable_row_monitor:
            # Verified plumbing: LLM(**kwargs) forwards this straight into
            # EngineArgs.enable_row_monitor (vllm/engine/arg_utils.py:601),
            # which — when enable_steering is set — reaches
            # SteeringConfig(enable_row_monitor=...) (arg_utils.py:2219) and
            # is a compute_hash factor that resizes the per-row probe-table
            # buffers. The activation assertion re-checks it actually landed
            # via status["row_monitor"]["enabled"].
            kwargs["enable_row_monitor"] = True

    llm = LLM(**kwargs)
    try:
        for _ in range(args.warmup):
            llm.generate(prompts, sp_list)
        # Activation gate — AFTER warmup, BEFORE the timed region (no added
        # work inside timing). Counters/tier/monitor state persist post-run.
        arm_diag = _assert_arm_active(args.arm, llm, args.batch_size)

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
        "steer_layers": layers,
        "arm_active": arm_diag,
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


def _cell_subprocess(
    args: argparse.Namespace, arm: str, batch_size: int
) -> dict[str, Any]:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--arm", arm, "--batch-size", str(batch_size),
        "--model", args.model, "--num-model-layers", str(args.num_model_layers),
        "--steer-layers", args.steer_layers, "--steer-hooks", args.steer_hooks,
        "--monitor-layer", str(args.monitor_layer),
        "--monitor-hook", args.monitor_hook,
        "--norm", str(args.norm), "--prompt-len", str(args.prompt_len),
        "--output-len", str(args.output_len), "--warmup", str(args.warmup),
        "--iters", str(args.iters), "--gpu-mem-util", str(args.gpu_mem_util),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        return {"arm": arm, "batch_size": batch_size, "error": f"exit {proc.returncode}"}
    for line in proc.stdout.splitlines():
        if line.startswith("CELL_RESULT "):
            return json.loads(line[len("CELL_RESULT "):])
    return {"arm": arm, "batch_size": batch_size, "error": "no CELL_RESULT emitted"}


def _detect_vllm_commit() -> str | None:
    """Best-effort HEAD of the vLLM package's source tree."""
    try:
        import vllm

        vdir = Path(vllm.__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(vdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--num-model-layers", type=int, default=34,
                        help="decoder layer count (for spread:N resolution)")
    parser.add_argument("--steer-layers", default="17",
                        help="comma ints, or 'spread:N' = N evenly spaced layers")
    parser.add_argument("--steer-hooks", default="post_block",
                        help="comma of hook names (pre_attn/post_attn/post_block)")
    parser.add_argument("--monitor-layer", type=int, default=-1,
                        help="monitor site layer (default: first steer layer)")
    parser.add_argument("--monitor-hook", default="",
                        help="monitor site hook (default: first steer hook)")
    parser.add_argument("--norm", type=float, default=8.0, help="steer vector L2 norm")
    parser.add_argument("--batch-sizes", default="1,8,16,32")
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--gpu-mem-util", type=float, default=0.92)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable cudagraphs (default: cudagraphs on)")
    parser.add_argument("--arms", default="", help="comma-subset of arms (default: all)")
    parser.add_argument("--vllm-commit", default="",
                        help="vLLM commit hash recorded in results "
                             "(default: auto-detect from the installed tree)")
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

    vllm_commit = args.vllm_commit or _detect_vllm_commit()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    arms = [a for a in ARM_ORDER if a in set(args.arms.split(","))] if args.arms else ARM_ORDER
    resolved_layers = resolve_layers(args.steer_layers, args.num_model_layers)

    print(f"Dynamic-steering benchmark: {args.model}")
    print(f"  arms={arms}")
    print(f"  batch_sizes={batch_sizes} prompt_len={args.prompt_len} "
          f"output_len={args.output_len} enforce_eager={args.enforce_eager}")
    print(f"  steer_layers={args.steer_layers} -> {resolved_layers} "
          f"hooks={args.steer_hooks}")
    print(f"  vllm_commit={vllm_commit}")
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
                    "steer_layers_spec": args.steer_layers,
                    "steer_layers": resolved_layers,
                    "steer_hooks": args.steer_hooks,
                    "monitor_layer": args.monitor_layer,
                    "monitor_hook": args.monitor_hook,
                    "num_sites": len(resolved_layers) * len(
                        [h for h in args.steer_hooks.split(",") if h.strip()]),
                    "norm": args.norm,
                    "enforce_eager": args.enforce_eager,
                    "warmup": args.warmup, "iters": args.iters,
                    "vllm_commit": vllm_commit,
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
