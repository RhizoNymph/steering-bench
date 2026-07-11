#!/usr/bin/env python3
"""Standalone capture-consumer profiling harness (vllm-direct).

Profiles the vLLM fork's capture pipeline overhead using ``logging_max`` as the
test case. Talks straight to ``vllm.LLM`` (no steering_bench dependency) so it
runs on the fork's own venv and puts the profiler as close to the code under
test as possible.

Configs (post_block hooks):
  baseline            no capture consumers (reference)
  logging_minimal     [mid] layer x last_prompt, WARNING
  logging_layers      ALL layers x last_prompt  (isolates the layer-breadth axis)
  logging_positions   [mid] layer x ALL pos     (isolates the position-count axis)
  logging_max         ALL layers x ALL pos, WARNING  (emits one log line per key)
  logging_max_silent  ALL layers x ALL pos, DEBUG     (level below threshold, so
                      logger.log short-circuits: same gather+dispatch machinery
                      but no string-format / log-I/O -> isolates pipeline cost)

Modes:
  plain   warmup + timed generate loop; prints tok/s. Wrap the whole process
          externally with nsys / py-spy / ncu.
  torch   same, but brackets the timed region with the built-in *worker*
          torch.profiler (enabled via ProfilerConfig), the only way to see the
          GPU work that runs in the V1 EngineCore subprocess. Writes a chrome
          trace into --trace-dir. An NVTX range (timed_region) is emitted for
          nsys scoping in either mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- config table ----------------------------------------------------------
NUM_LAYERS_DEFAULT = 34
MID_LAYER = 6


def _consumer(positions: str, level: str, layers: list[int]) -> dict:
    return {
        "name": "logging",
        "params": {
            "hooks": {"post_block": layers},
            "positions": positions,
            "level": level,
        },
    }


def build_config(name: str, num_layers: int) -> list[dict] | None:
    all_layers = list(range(num_layers))
    mid = [min(MID_LAYER, num_layers - 1)]
    table = {
        "baseline": None,
        "logging_minimal": _consumer("last_prompt", "WARNING", mid),
        "logging_layers": _consumer("last_prompt", "WARNING", all_layers),
        "logging_positions": _consumer("all", "WARNING", mid),
        "logging_max": _consumer("all", "WARNING", all_layers),
        "logging_max_silent": _consumer("all", "DEBUG", all_layers),
    }
    if name not in table:
        raise SystemExit(f"unknown config {name!r}; choices: {sorted(table)}")
    c = table[name]
    return None if c is None else [c]


def make_prompts(batch: int, approx_tokens: int) -> list[str]:
    # Distinct prompts (unique leading index) so prefix caching cannot collapse
    # the batch and hide real per-request capture work. ~approx_tokens words.
    base = " ".join(["benchmark"] * max(1, approx_tokens - 4))
    return [f"{i} sample request {base}" for i in range(batch)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--config", default="logging_max")
    ap.add_argument("--mode", choices=["plain", "torch"], default="plain")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    trace_dir_abs = ""
    if args.mode == "torch":
        tdir = args.trace_dir or f"./trace_{args.config}"
        Path(tdir).mkdir(parents=True, exist_ok=True)
        trace_dir_abs = os.path.abspath(tdir)
        # keep the trace small / focused
        args.iters = min(args.iters, 2)

    import torch
    from vllm import LLM, SamplingParams

    consumers = build_config(args.config, args.num_layers)

    load_opts: dict = {
        "model": args.model,
        "enable_steering": False,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        # distinct prompts + no cache reuse -> honest per-request capture cost
        "enable_prefix_caching": False,
    }
    if consumers is not None:
        load_opts["capture_consumers"] = consumers
    if args.mode == "torch":
        # The fork enables the built-in worker profiler via a structured
        # ProfilerConfig; with_stack is on by default so the trace carries
        # Python frames for CPU attribution.
        from vllm.config.profiler import ProfilerConfig

        load_opts["profiler_config"] = ProfilerConfig(
            profiler="torch", torch_profiler_dir=trace_dir_abs
        )

    print(f"[{args.config}] loading (mode={args.mode})...", flush=True)
    llm = LLM(**load_opts)

    prompts = make_prompts(args.batch_size, args.prompt_len)
    sp = SamplingParams(max_tokens=args.output_len, temperature=0.0)

    def run() -> None:
        llm.generate(prompts, sp, use_tqdm=False)

    print(f"[{args.config}] warmup x{args.warmup}...", flush=True)
    for _ in range(args.warmup):
        run()

    if args.mode == "torch" and hasattr(llm, "start_profile"):
        llm.start_profile()

    # NVTX range so nsys can scope to steady-state (excludes model load).
    torch.cuda.nvtx.range_push("timed_region")
    samples: list[float] = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("gen_iter")
        run()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    torch.cuda.nvtx.range_pop()

    if args.mode == "torch" and hasattr(llm, "stop_profile"):
        llm.stop_profile()
        time.sleep(2)  # let the worker flush the trace

    mean_ms = sum(samples) / len(samples)
    toks = args.batch_size * args.output_len
    tps = toks / (mean_ms / 1000.0)
    out = {
        "config": args.config,
        "mode": args.mode,
        "label": args.label,
        "batch_size": args.batch_size,
        "output_len": args.output_len,
        "iters": args.iters,
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(min(samples), 2),
        "tokens_per_sec": round(tps, 1),
    }
    print("RESULT " + json.dumps(out), flush=True)
    if args.mode == "torch":
        print(f"[trace] {trace_dir_abs}", flush=True)
    sys.stdout.flush()

    # Graceful shutdown so the V1 EngineCore subprocess releases VRAM before the
    # next config starts (a hard os._exit orphans it, holding several GiB). A
    # daemon watchdog hard-exits if normal teardown hangs, so a loop can't stall.
    import threading

    def _watchdog() -> None:
        time.sleep(45)
        os._exit(0)

    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    os._exit(0)


if __name__ == "__main__":
    main()
