#!/usr/bin/env python3
"""Static-steering cudagraph-overhead probe — branch-portable.

Isolates ONE question: does enabling per-request *static* steering (the
pre-dynamic ``apply_steering`` gather-and-add, driven purely through
``SamplingParams.steering_vectors``) inflate ``generate()`` time under
cudagraphs at decode batch > 16? This is the minimal, self-contained
reproducer used to check whether the bs>16 cudagraph steering spike seen
in bench_dynamic_steering.py is pre-existing on feat/integration or was
introduced by the dynamic-steering machinery.

Deliberately imports NOTHING from steering_bench or the dynamic-steering
action-queue/consumer APIs — only ``vllm`` + numpy + stdlib — so the same
file runs unchanged on feat/integration and feat/dynamic-steering. Drive it
with each branch's venv python.

Two arms, fresh subprocess per cell:
  off    enable_steering=False, no steering        (cold baseline)
  steer  enable_steering=True + every request carries a base-phase
         steering vector at (hook, layer) -> apply_steering active on
         prefill AND decode (base phase dodges the decode-only path)

    VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/bench_static_steering.py \
      --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf --layer 30 --hidden 5376 \
      --batch-sizes 16,24,32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def _gpu_clock_mhz() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:
        return None


def _stats(samples):
    import numpy as np
    a = np.array(samples)
    return float(np.mean(a)), float(np.percentile(a, 90))


def _run_cell(args) -> dict:
    import numpy as np
    import torch

    from vllm import LLM, SamplingParams

    steer = args.arm == "steer"
    prompts = [" ".join(["hello"] * max(1, int(args.prompt_len / 1.3)))] * args.batch_size

    sp_kwargs = dict(max_tokens=args.output_len, temperature=0.0, seed=0)
    if steer:
        rng = np.random.default_rng(1)
        v = rng.standard_normal(args.hidden).astype(np.float32)
        v = v / float(np.linalg.norm(v)) * args.norm
        # SteeringVectorSpec: {hook: {layer: [floats]}}, base phase (prefill+decode).
        sp_kwargs["steering_vectors"] = {args.hook: {args.layer: v.tolist()}}
    sp = SamplingParams(**sp_kwargs)
    sp_list = [sp] * args.batch_size

    kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.prompt_len + args.output_len + 64,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    if steer:
        kwargs["enable_steering"] = True

    llm = LLM(**kwargs)
    try:
        for _ in range(args.warmup):
            llm.generate(prompts, sp_list)
        samples = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts, sp_list)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        clk = _gpu_clock_mhz()
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    mean_ms, p90_ms = _stats(samples)
    return {
        "arm": args.arm, "batch_size": args.batch_size,
        "mean_ms": mean_ms, "p90_ms": p90_ms,
        "tokens_per_sec": (args.batch_size * args.output_len) / (mean_ms / 1000.0),
        "gpu_clock_mhz": clk,
    }


def _cell_subprocess(args, arm, bs) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--cell",
           "--arm", arm, "--batch-size", str(bs), "--model", args.model,
           "--layer", str(args.layer), "--hidden", str(args.hidden),
           "--norm", str(args.norm), "--prompt-len", str(args.prompt_len),
           "--output-len", str(args.output_len), "--warmup", str(args.warmup),
           "--iters", str(args.iters), "--gpu-mem-util", str(args.gpu_mem_util),
           "--hook", args.hook]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return {"arm": arm, "batch_size": bs, "error": f"exit {proc.returncode}"}
    for line in proc.stdout.splitlines():
        if line.startswith("CELL_RESULT "):
            return json.loads(line[len("CELL_RESULT "):])
    return {"arm": arm, "batch_size": bs, "error": "no CELL_RESULT"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--hidden", type=int, default=5376, help="model hidden size")
    p.add_argument("--hook", default="post_mlp")
    p.add_argument("--norm", type=float, default=8.0)
    p.add_argument("--batch-sizes", default="16,24,32")
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", help=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.cell:
        print("CELL_RESULT " + json.dumps(_run_cell(args)), flush=True)
        return

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    print(f"Static-steering cudagraph probe: {args.model}")
    print(f"  layer={args.layer} hidden={args.hidden} batch_sizes={batch_sizes} "
          f"enforce_eager={args.enforce_eager}\n")
    rows = []
    for bs in batch_sizes:
        base = None
        for arm in ("off", "steer"):
            r = _cell_subprocess(args, arm, bs)
            if "error" in r:
                print(f"  bs={bs} [{arm}] ERROR {r['error']}")
                rows.append(r)
                continue
            if arm == "off":
                base = r["mean_ms"]
            ov = (r["mean_ms"] - base) / base * 100.0 if base else None
            r["overhead_pct"] = ov
            ovs = f"{ov:+.1f}%" if ov is not None else "baseline"
            print(f"  bs={bs:>3} [{arm:<5}] mean={r['mean_ms']:.1f}ms "
                  f"tps={r['tokens_per_sec']:.0f} {ovs} clk={r.get('gpu_clock_mhz')}MHz")
            rows.append(r)
    print("\nRESULTS_JSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
