#!/usr/bin/env python3
"""End-to-end capture consumer overhead benchmark.

Measures the wall-clock throughput overhead of enabling capture consumers
during real LLM.generate() calls with facebook/opt-125m (or another model).
Each configuration constructs a fresh LLM, warms up, then runs timed
generate() calls.

Consumer configurations benchmarked:
  baseline           — no consumers (reference)
  logging_minimal    — 1 logging consumer, last_prompt, layer 6
  logging_max        — 1 logging consumer, all positions, all layers
  logging_3x         — 3 logging consumers on same hook/layer (union-gather path)
  filesystem_minimal — 1 filesystem consumer, last_prompt, layer 6
                       (requires per-request SamplingParams.capture)
  driver_minimal     — 1 driver RecordingConsumer, last_prompt, layer 6

Metrics reported per config × batch_size:
  tokens_per_sec    — (batch_size × output_len) / mean_generate_s
  mean_ms           — mean wall-clock ms per generate() call
  overhead_pct      — % overhead vs baseline (positive = slower)
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.capture_consumers.consumers import RecordingDriverConsumer
from steering_bench.capture_consumers.runner import get_model_config, make_prompts
from steering_bench.engine.base import SteeringConfig
from steering_bench.engine.capture import CaptureConsumerSpec
from steering_bench.engine.engines.vllm import VllmSteeringEngine
from steering_bench.engine.spec import GenerationRequest, RequestCapture
from steering_bench.output import write_result
from steering_bench.timing import compute_stats

# Config = (name, capture-consumer specs, optional per-request capture opt-in).
CaptureConfig = tuple[str, list[CaptureConsumerSpec], RequestCapture | None]


def _measure(
    engine: VllmSteeringEngine,
    requests: list[GenerationRequest],
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        engine.generate(requests)

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(requests)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    return samples


def _run_config(
    model: str,
    config_name: str,
    specs: list[CaptureConsumerSpec],
    request_capture: RequestCapture | None,
    batch_size: int,
    prompt_len: int,
    output_len: int,
    warmup: int,
    iters: int,
) -> dict:
    prompts = make_prompts(batch_size, prompt_len, model=model)
    requests = [
        GenerationRequest(prompt=p, max_tokens=output_len, capture=request_capture)
        for p in prompts
    ]

    print(f"    [{config_name}] loading model...", flush=True)
    engine = VllmSteeringEngine()
    engine.configure_capture(specs)
    # Capture-only benchmark: keep steering off so the baseline is not inflated
    # by the steering subsystem.
    engine.load(
        model,
        steering_config=SteeringConfig(enable_steering=False),
        max_model_len=512,
    )

    print(f"    [{config_name}] warmup={warmup}, iters={iters}...", flush=True)
    try:
        samples = _measure(engine, requests, warmup, iters)
        stats = compute_stats(samples)
        tokens_per_sec = (batch_size * output_len) / (stats.mean_ms / 1000.0)
        return {
            "config": config_name,
            "mean_ms": stats.mean_ms,
            "median_ms": stats.median_ms,
            "p90_ms": stats.p90_ms,
            "tokens_per_sec": tokens_per_sec,
        }
    except torch.cuda.OutOfMemoryError:
        return {"config": config_name, "error": "OOM"}
    finally:
        engine.teardown()
        gc.collect()
        torch.cuda.empty_cache()


def _logging_spec(mid_layer: int) -> CaptureConsumerSpec:
    return CaptureConsumerSpec(
        name="logging",
        params={"hooks": {"post_block": [mid_layer]}, "positions": "last_prompt", "level": "WARNING"},
    )


def _build_configs(
    model: str,
    model_cfg: dict,
    tmpdir: str,
) -> list[CaptureConfig]:
    """Return list of (config_name, consumer specs, per-request capture)."""
    num_layers = model_cfg["num_layers"]
    mid_layer = min(6, num_layers - 1)
    all_layers = list(range(num_layers))

    driver_consumer = RecordingDriverConsumer(
        hooks={"post_block": [mid_layer]},
        positions="last_prompt",
    )

    return [
        ("baseline", [], None),
        ("logging_minimal", [_logging_spec(mid_layer)], None),
        ("logging_max", [
            CaptureConsumerSpec(
                name="logging",
                params={"hooks": {"post_block": all_layers}, "positions": "all", "level": "WARNING"},
            )
        ], None),
        ("logging_3x_same_hook", [
            CaptureConsumerSpec(
                name="logging",
                instance_name=f"log_{c}",
                params={"hooks": {"post_block": [mid_layer]}, "positions": "last_prompt", "level": "WARNING"},
            )
            for c in ("a", "b", "c")
        ], None),
        ("filesystem_minimal", [
            CaptureConsumerSpec(name="filesystem", params={"root": tmpdir, "writer_threads": 4})
        ], RequestCapture(
            consumer="filesystem",
            hooks={"post_block": [mid_layer]},
            positions="last_prompt",
            tag="benchmark",
        )),
        ("driver_minimal", [
            CaptureConsumerSpec(name="driver_recording", location="driver", instance=driver_consumer)
        ], None),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM.generate() throughput with/without capture consumers"
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument(
        "--batch-sizes", default="1,8,32",
        help="Comma-separated batch sizes"
    )
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated subset of config names to run (default: all)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for E2E capture benchmark")
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    model_cfg = get_model_config(args.model)

    print(f"Capture E2E benchmark: {args.model}")
    print(
        f"  batch_sizes={batch_sizes}, output_len={args.output_len}, "
        f"prompt_len={args.prompt_len}"
    )
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        all_configs = _build_configs(args.model, model_cfg, tmpdir)

        if args.configs:
            wanted = set(args.configs.split(","))
            all_configs = [cfg for cfg in all_configs if cfg[0] in wanted]

        all_results = []

        for batch_size in batch_sizes:
            print(f"\n--- batch_size={batch_size} ---")
            baseline_mean: float | None = None

            for config_name, specs, request_capture in all_configs:
                result = _run_config(
                    model=args.model,
                    config_name=config_name,
                    specs=specs,
                    request_capture=request_capture,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    output_len=args.output_len,
                    warmup=args.warmup,
                    iters=args.iters,
                )

                if "error" in result:
                    print(f"    [{config_name}] {result['error']}")
                    all_results.append({
                        "batch_size": batch_size,
                        "config": config_name,
                        "error": result["error"],
                    })
                    continue

                if config_name == "baseline":
                    baseline_mean = result["mean_ms"]

                overhead_pct = None
                if baseline_mean is not None and baseline_mean > 0:
                    overhead_pct = (
                        (result["mean_ms"] - baseline_mean) / baseline_mean * 100.0
                    )

                result["overhead_pct"] = overhead_pct
                overhead_str = (
                    f"{overhead_pct:+.1f}%" if overhead_pct is not None else "baseline"
                )
                print(
                    f"    [{config_name}] "
                    f"mean={result['mean_ms']:.1f}ms  "
                    f"tps={result['tokens_per_sec']:.0f}  "
                    f"overhead={overhead_str}"
                )

                write_result(
                    benchmark="capture.e2e",
                    parameters={
                        "model": args.model,
                        "config": config_name,
                        "batch_size": batch_size,
                        "prompt_len": args.prompt_len,
                        "output_len": args.output_len,
                        "warmup": args.warmup,
                        "iters": args.iters,
                    },
                    results={k: v for k, v in result.items() if k != "config"},
                    output_dir=args.output_dir,
                    tag=args.tag,
                )

                all_results.append({"batch_size": batch_size, **result})

    # Summary table
    print(f"\n{'=' * 105}")
    print(f"  Capture E2E Benchmark: {args.model}")
    print(f"{'=' * 105}")
    print(
        f"{'batch':>6} {'config':<26} {'mean_ms':>10} {'p90_ms':>10} "
        f"{'tps':>10} {'overhead':>10}"
    )
    print("-" * 105)
    for r in all_results:
        if "error" in r:
            print(f"{r['batch_size']:>6} {r['config']:<26} {'ERROR':>10}")
            continue
        overhead = r.get("overhead_pct")
        overhead_str = f"{overhead:+.1f}%" if overhead is not None else "baseline"
        print(
            f"{r['batch_size']:>6} {r['config']:<26} "
            f"{r['mean_ms']:>10.1f} {r['p90_ms']:>10.1f} "
            f"{r['tokens_per_sec']:>10.0f} {overhead_str:>10}"
        )
    print(f"{'=' * 105}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
