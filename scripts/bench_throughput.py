#!/usr/bin/env python3
"""vLLM system benchmark: batch throughput with varying steering configs.

Measures total tokens/sec when processing batches with 0/1/4/8
distinct steering configurations.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import (
    even_layer_subset,
    random_steering_vectors,
    random_steering_vectors_diverse,
)

NAMED_BENCH_MODULE = "bench_named_shared"


def _vectors_to_named_payload(vectors: dict) -> dict:
    """Convert random_steering_vectors output to register_steering_modules payload."""
    import numpy as np

    coerced: dict = {}
    for hook, layer_dict in vectors.items():
        coerced[hook] = {}
        for layer_idx, entry in layer_dict.items():
            if isinstance(entry, np.ndarray):
                entry = entry.tolist()
            coerced[hook][layer_idx] = entry
    return {"vectors": coerced}

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
}


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def _legacy_mode_for_distinct(distinct_configs: int) -> str:
    """Translate a legacy ``--configs-sweep`` integer to a mode name.

    ``0`` → ``disabled`` (steering subsystem off).
    ``1`` → ``inline_shared`` (single spec across the batch).
    ``N`` → ``per_request_N`` (cycle ``N`` distinct specs).
    """
    if distinct_configs <= 0:
        return "disabled"
    if distinct_configs == 1:
        return "inline_shared"
    return f"per_request_{distinct_configs}"


def run_throughput(
    model: str,
    num_prompts: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
    mode: str | None = None,
    distinct_configs: int | None = None,
    max_steering_configs_override: int | None = None,
    enable_prefix_caching: bool = True,
    num_hooks: int = 1,
    num_layers_steered: int | None = None,
) -> dict:
    """Run throughput benchmark for a single mode.

    Either *mode* or *distinct_configs* must be set.  If only
    *distinct_configs* is provided, it is translated to a mode via
    :func:`_legacy_mode_for_distinct` for backwards compat with the
    historical CLI.

    Mode catalog matches :func:`bench_latency.run_mode`:
    ``disabled``, ``enabled_idle``, ``inline_shared``, ``inline_unique``,
    ``named_shared``, ``per_request_N``.
    """
    from vllm import LLM, SamplingParams

    if mode is None:
        if distinct_configs is None:
            raise ValueError("Either mode or distinct_configs must be set")
        mode = _legacy_mode_for_distinct(distinct_configs)

    enable_steering = mode != "disabled"

    if max_steering_configs_override is not None:
        max_steering = max_steering_configs_override
    elif mode == "inline_unique":
        # Each request fresh — the worker table needs ≥ unique-per-iter
        # plus headroom for warmup vs timed iters.
        max_steering = max(64, num_prompts * 2)
    elif mode.startswith("per_request_"):
        suffix = mode.split("_")[-1]
        try:
            n = int(suffix)
        except ValueError:
            n = 4
        # Active table needs to hold both the inline-packed first sights
        # AND the auto-promoted named-resolved copies of each spec (same
        # vectors, separate table slots until first-sight requests drain),
        # so 2N at minimum.  Add headroom for in-flight transitions and
        # for the disabled-mode max=4 floor.
        max_steering = max(n * 4, 16)
    else:
        max_steering = 4

    all_hooks = ["post_block", "post_attn", "pre_attn"]
    if num_hooks < 1 or num_hooks > len(all_hooks):
        raise ValueError(
            f"num_hooks must be in [1, {len(all_hooks)}], got {num_hooks}"
        )
    active_hooks = all_hooks[:num_hooks]
    layer_subset = (
        None if num_layers_steered is None
        else even_layer_subset(num_layers, num_layers_steered)
    )

    print(f"    Loading model (mode={mode}, max_configs={max_steering}, "
          f"prefix_cache={enable_prefix_caching}, hooks={active_hooks}, "
          f"layers_steered={len(layer_subset) if layer_subset else num_layers})...",
          flush=True)
    llm = LLM(
        model=model,
        enable_steering=enable_steering,
        max_steering_configs=max_steering,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    prompts = make_prompts(num_prompts, prompt_len)

    if mode == "disabled" or mode == "enabled_idle":
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        sp_list = [sp] * num_prompts

    elif mode == "inline_shared":
        vectors = random_steering_vectors(
            hidden_size=hidden_size, num_layers=num_layers,
            hook_points=active_hooks, scale=0.1, seed=42,
            layer_subset=layer_subset,
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_vectors=vectors,
        )
        sp_list = [sp] * num_prompts

    elif mode == "inline_unique":
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size, num_layers=num_layers,
            num_configs=num_prompts, hook_points=active_hooks, scale=0.1,
            base_seed=42, layer_subset=layer_subset,
        )
        sp_list = [
            SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                steering_vectors=diverse[i],
            )
            for i in range(num_prompts)
        ]

    elif mode == "named_shared":
        vectors = random_steering_vectors(
            hidden_size=hidden_size, num_layers=num_layers,
            hook_points=active_hooks, scale=0.1, seed=42,
            layer_subset=layer_subset,
        )
        llm.llm_engine.collective_rpc(
            "register_steering_modules",
            kwargs={
                "modules": {NAMED_BENCH_MODULE: _vectors_to_named_payload(vectors)},
                "replace": True,
            },
        )
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
            steering_module_ref=(NAMED_BENCH_MODULE, 1.0),
        )
        sp_list = [sp] * num_prompts

    elif mode.startswith("per_request_"):
        suffix = mode.split("_")[-1]
        try:
            distinct = int(suffix)
        except ValueError:
            raise ValueError(f"Unknown mode: {mode}")
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size, num_layers=num_layers,
            num_configs=distinct, hook_points=active_hooks, scale=0.1,
            base_seed=42, layer_subset=layer_subset,
        )
        sp_list = [
            SamplingParams(
                max_tokens=max_tokens, temperature=0.0,
                steering_vectors=diverse[i % distinct],
            )
            for i in range(num_prompts)
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Warmup
    print(f"    Warmup ({warmup} iters)...", flush=True)
    for _ in range(warmup):
        llm.generate(prompts, sp_list)

    # Measure
    print(f"    Measuring ({iters} iters)...", flush=True)
    samples_ms = []
    total_output_tokens_list = []

    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_list)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        samples_ms.append(elapsed_ms)

        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_output_tokens_list.append(total_out)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Compute throughput
    avg_output_tokens = sum(total_output_tokens_list) / len(total_output_tokens_list)
    total_tokens_per_iter = num_prompts * prompt_len + avg_output_tokens  # input + output

    latency_stats = compute_stats(samples_ms)
    throughput_samples = [
        total_tokens_per_iter / (ms / 1000.0) for ms in samples_ms
    ]
    throughput_stats = compute_stats(throughput_samples)

    # Rename throughput stat keys from *_ms (misleading since the unit is
    # tokens/sec, not milliseconds) to *_tps.
    throughput_dict = {
        k.replace("_ms", "_tps"): v
        for k, v in throughput_stats.to_dict().items()
        if k != "samples_ms"
    }

    return {
        "latency_ms": {k: v for k, v in latency_stats.to_dict().items() if k != "samples_ms"},
        "throughput_tokens_per_sec": throughput_dict,
        "avg_output_tokens": avg_output_tokens,
        "total_tokens_per_iter": total_tokens_per_iter,
        "samples_ms": samples_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM throughput with steering")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--configs-sweep",
        default=None,
        help="Legacy. Comma-separated distinct_configs counts (0/1/4/8) "
             "translated to disabled/inline_shared/per_request_N modes.  "
             "Prefer --modes for new runs.",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated mode list.  See bench_latency.run_mode for "
             "the catalog (disabled, enabled_idle, inline_shared, "
             "inline_unique, named_shared, per_request_N).  Mutually "
             "exclusive with --configs-sweep.",
    )
    parser.add_argument(
        "--num-hooks",
        type=int,
        default=1,
        help="Number of hook points to populate per request (1..3).",
    )
    parser.add_argument(
        "--num-layers-steered",
        type=int,
        default=None,
        help="Number of layers to steer (defaults to all model layers).",
    )
    parser.add_argument(
        "--max-steering-configs",
        type=int,
        default=None,
        help="Override auto-computed max_steering_configs. "
             "Default = mode-derived (≥ batch×2 for inline_unique).",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching (to isolate its effect on throughput).",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.modes and args.configs_sweep:
        parser.error("Pass exactly one of --modes / --configs-sweep")
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    elif args.configs_sweep:
        modes = [
            _legacy_mode_for_distinct(int(x))
            for x in args.configs_sweep.split(",")
        ]
    else:
        # Default sweep covers the new modes plus the original config sweep.
        modes = [
            "disabled",
            "enabled_idle",
            "inline_shared",
            "inline_unique",
            "named_shared",
            "per_request_4",
        ]
    config = MODEL_CONFIGS.get(args.model, {"hidden_size": 2560, "num_layers": 34})

    print(f"Throughput benchmark: {args.model}")
    print(f"Prompts: {args.num_prompts}, prompt_len: {args.prompt_len}, max_tokens: {args.max_tokens}")
    print(f"Modes: {modes}")
    print()

    all_results = []
    baseline_throughput = None

    for mode in modes:
        print(f"\n--- mode={mode} ---")

        try:
            result = run_throughput(
                model=args.model,
                num_prompts=args.num_prompts,
                prompt_len=args.prompt_len,
                max_tokens=args.max_tokens,
                mode=mode,
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                max_steering_configs_override=args.max_steering_configs,
                enable_prefix_caching=not args.disable_prefix_cache,
                num_hooks=args.num_hooks,
                num_layers_steered=args.num_layers_steered,
            )

            mean_tps = result["throughput_tokens_per_sec"]["mean_tps"]
            mean_latency = result["latency_ms"]["mean_ms"]
            print(f"    throughput: {mean_tps:.0f} tokens/sec")
            print(f"    batch latency: {mean_latency:.0f} ms")

            if mode == "disabled":
                baseline_throughput = mean_tps

            overhead_pct = None
            if baseline_throughput and baseline_throughput > 0:
                overhead_pct = (baseline_throughput - mean_tps) / baseline_throughput * 100
                print(f"    throughput loss vs baseline: {overhead_pct:.1f}%")

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM!")
            result = {"error": "OOM"}
            overhead_pct = None

        params = {
            "model": args.model,
            "mode": mode,
            "max_steering_configs_override": args.max_steering_configs,
            "num_prompts": args.num_prompts,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "prefix_caching": not args.disable_prefix_cache,
            "num_hooks": args.num_hooks,
            "num_layers_steered": (
                args.num_layers_steered
                if args.num_layers_steered is not None
                else config["num_layers"]
            ),
        }
        results_out = {k: v for k, v in result.items() if k != "samples_ms"}
        if overhead_pct is not None:
            results_out["throughput_loss_pct"] = overhead_pct

        write_result(
            benchmark="vllm.throughput",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=result.get("samples_ms"),
        )
        all_results.append({
            "mode": mode,
            "results": results_out,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  Throughput Benchmark Summary: {args.model}")
    print(f"{'=' * 80}")
    print(f"{'mode':>20} {'tokens/sec':>14} {'batch_ms':>12} {'loss':>10}")
    print(f"{'-' * 80}")
    for r in all_results:
        res = r["results"]
        if "error" in res:
            print(f"{r['mode']:>20} {'OOM':>14}")
            continue
        tps = res["throughput_tokens_per_sec"]["mean_tps"]
        lat = res["latency_ms"]["mean_ms"]
        loss = res.get("throughput_loss_pct")
        loss_str = f"{loss:.1f}%" if loss is not None else "baseline"
        print(f"{r['mode']:>20} {tps:>14.0f} {lat:>12.0f} {loss_str:>10}")
    print(f"{'=' * 80}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
