#!/usr/bin/env python3
"""Online serving benchmark: TTFT, TPOT, ITL, E2EL across steering modes.

Thin driver over the :class:`ServingEngine` seam. Launches the vLLM
OpenAI-compatible API server as a subprocess and drives it through streaming
completions, measuring time-to-first-token (TTFT), time-per-output-token (TPOT),
inter-token latency (ITL), and end-to-end latency (E2EL) per steering mode.

As of Phase 5 the subprocess launch, per-request vector packing, named-module
registration, and the timing-dump endpoint are all owned by the adapter
(``steering_bench.engine.serving`` / ``engine.engines.vllm_serving``); this script
only parses flags and hands them to :class:`ServingBenchmark`. Equivalent CLI:
``python -m steering_bench run serving --engine vllm [flags]``.

Modes:
    disabled             server started without --enable-steering
    enabled_idle         steering enabled, no vectors in requests
    all_steered_shared   every request uses the same steering vector
    named_shared         one module registered, referenced by name per request
    per_request_n4       4 distinct configs spread across requests
    per_request_n16      16 distinct configs spread across requests

Workloads:
    synthetic (default)  fixed-length prompts generated locally
    sharegpt             via --sharegpt-path pointing at a local ShareGPT_V3 json
"""

from __future__ import annotations

import argparse

from steering_bench.engine.engines.vllm_serving import VllmServingEngine
from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.harness.benchmarks.serving import ServingBenchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Online serving benchmark")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/serving/")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--hook", default="post_block")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    ServingBenchmark.add_args(parser)
    args = parser.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        warmup=0,
        iters=0,
        max_tokens=args.max_tokens,
        layer=0,
        hook=args.hook,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    options = ServingBenchmark.options_from_args(args)
    engine = VllmServingEngine(python_bin=args.python_bin)
    ServingBenchmark(engine, config, **options).run()


if __name__ == "__main__":
    main()
