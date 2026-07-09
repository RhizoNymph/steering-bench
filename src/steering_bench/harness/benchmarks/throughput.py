"""Throughput benchmark: tokens/sec for a steering mode, any engine.

The engine-agnostic successor to the offline throughput logic in
``scripts/bench_throughput.py``: run a batch of ``--num-prompts`` requests under
one steering mode and report total (input + output) tokens/sec.  Mode selection,
load-time :class:`SteeringConfig`, and named-module registration are shared with
the latency benchmark via :class:`ModeBenchmark`; sweeping several modes / batch
sizes is done by repeated CLI invocations (the ``scripts/bench_throughput.py``
matrix remains for fork-specific deep-dives).
"""

from __future__ import annotations

from typing import Any

from steering_bench.harness.benchmarks.modes import ModeBenchmark
from steering_bench.timing import TimingStats, compute_stats


class ThroughputBenchmark(ModeBenchmark):
    """Batch throughput (tokens/sec) for the chosen engine + mode."""

    benchmark_name = "harness.throughput"
    batch_flag = "--num-prompts"
    batch_dest = "num_prompts"
    default_batch = 64

    def extra_results(
        self, stats: TimingStats, avg_output_tokens: float, num_requests: int
    ) -> dict[str, Any]:
        # Throughput counts input + output tokens processed per second, matching
        # scripts/bench_throughput.py.  With one shared prompt of ~prompt_len
        # tokens per request, input tokens ~= num_requests * prompt_len.
        if stats.mean_ms <= 0 or not stats.samples_ms:
            return {}
        input_tokens = num_requests * self.prompt_len
        total_tokens_per_iter = input_tokens + avg_output_tokens
        throughput_samples = [
            total_tokens_per_iter / (ms / 1000.0) for ms in stats.samples_ms
        ]
        tput = compute_stats(throughput_samples)
        return {
            "throughput_tokens_per_sec": {
                k.replace("_ms", "_tps"): v
                for k, v in tput.to_dict().items()
                if k != "samples_ms"
            },
            "total_tokens_per_iter": total_tokens_per_iter,
        }
