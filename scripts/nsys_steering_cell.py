#!/usr/bin/env python3
"""Standalone single-config nsys target for the cudagraph steering spike.

Builds ONE LLM (off or static-steer), warms up (un-profiled), then runs the
measured generate() calls bracketed by ``cudaProfilerStart/Stop`` so an nsys
``--capture-range=cudaProfilerApi`` trace captures only steady state — no
model load / cudagraph capture noise. No torch.profiler inside (avoids
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS under nsys). Branch-portable: only uses
``vllm`` + ``SamplingParams.steering_vectors`` (no dynamic-steering APIs).

Usage (profile off, then steer; then diff the stats):

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,nvtx,osrt -o /tmp/nsys_off \
      python scripts/nsys_steering_cell.py --arm off --batch-size 24 \
        --model ~/Models/gemma-4-31B-it-Q4_K_S.gguf
    ... --arm static  ... -o /tmp/nsys_static   # 8-arg op (table gather)
    ... --arm dynamic ... -o /tmp/nsys_dynamic  # §5.4 tier via consumer
    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum /tmp/nsys_dynamic.nsys-rep
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # off      : no steering (baseline)
    # static   : SamplingParams.steering_vectors (table gather, 8-arg op)
    # dynamic  : §5.4 dynamic tier via the bench_steer_async consumer
    #            (per-step token_scales/dvec machinery) — needs steering-bench
    #            installed in the venv for the entry point.
    p.add_argument("--arm", choices=["off", "static", "dynamic"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--hidden", type=int, default=5376)
    p.add_argument("--hook", default="post_block")
    p.add_argument("--norm", type=float, default=8.0)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="override max_model_len (default prompt+output+64); "
                        "use to bound KV/cudagraph memory on large models")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="override max_num_seqs (caps KV/cudagraph reservation)")
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    args = p.parse_args()

    import numpy as np
    import torch
    from vllm import LLM, SamplingParams

    prompts = [" ".join(["hello"] * max(1, int(args.prompt_len / 1.3)))] * args.batch_size
    sp_kwargs = dict(max_tokens=args.output_len, temperature=0.0, seed=0)
    if args.arm == "static":
        v = np.random.default_rng(1).standard_normal(args.hidden).astype(np.float32)
        v = v / float(np.linalg.norm(v)) * args.norm
        sp_kwargs["steering_vectors"] = {args.hook: {args.layer: v.tolist()}}
    sp_list = [SamplingParams(**sp_kwargs)] * args.batch_size

    kwargs = dict(
        model=args.model, gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len or (args.prompt_len + args.output_len + 64),
        enforce_eager=args.enforce_eager, seed=0,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.arm in ("static", "dynamic"):
        kwargs["enable_steering"] = True
    if args.arm == "dynamic":
        kwargs["max_dynamic_steering_configs"] = 4
        kwargs["capture_consumers"] = [
            {"name": "bench_steer_async",
             "params": {"layer": args.layer, "norm": args.norm}}
        ]
    llm = LLM(**kwargs)

    for _ in range(args.warmup):
        llm.generate(prompts, sp_list)
    torch.cuda.synchronize()

    # Only this region is captured under --capture-range=cudaProfilerApi.
    torch.cuda.profiler.start()
    for _ in range(args.iters):
        llm.generate(prompts, sp_list)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"[nsys_steering_cell] arm={args.arm} bs={args.batch_size} done")


if __name__ == "__main__":
    main()
