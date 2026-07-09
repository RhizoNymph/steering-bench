"""Latency benchmark: per-iteration generation latency, any engine, any mode.

The engine-agnostic successor to the offline latency logic in
``scripts/bench_latency.py``: drive one of the steering modes
(``disabled`` / ``enabled_idle`` / ``inline_shared`` / ``inline_unique`` /
``named_shared`` / ``per_request_N``) through the engine seam and let
:meth:`Benchmark.run` time a batch of ``--batch-size`` requests.  ``named_shared``
registers a module once via ``engine.register_module`` and references it by name;
on engines without ``named_modules`` it degrades to inline-shared.
"""

from __future__ import annotations

from steering_bench.harness.benchmarks.modes import ModeBenchmark


class LatencyBenchmark(ModeBenchmark):
    """Single/batched steered-generation latency for the chosen engine + mode."""

    benchmark_name = "harness.latency"
    batch_flag = "--batch-size"
    batch_dest = "batch_size"
    default_batch = 1
