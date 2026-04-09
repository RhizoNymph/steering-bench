"""GPU-synchronized timing utilities.

All timing uses CUDA events for accurate GPU measurement,
with torch.cuda.synchronize() barriers to ensure completeness.
"""

from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TimingStats:
    """Aggregated timing statistics from a series of measurements."""

    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    stddev_ms: float
    p10_ms: float
    p25_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p99_ms: float
    n: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def compute_stats(samples_ms: list[float]) -> TimingStats:
    """Compute timing statistics from a list of millisecond measurements."""
    arr = np.array(samples_ms)
    return TimingStats(
        samples_ms=samples_ms,
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        stddev_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        p10_ms=float(np.percentile(arr, 10)),
        p25_ms=float(np.percentile(arr, 25)),
        p50_ms=float(np.percentile(arr, 50)),
        p75_ms=float(np.percentile(arr, 75)),
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        n=len(arr),
    )


def cuda_timer(warmup: int, iters: int, fn: callable, *args, **kwargs) -> TimingStats:
    """Time a GPU function using CUDA events.

    Args:
        warmup: Number of warmup iterations (not measured).
        iters: Number of measured iterations.
        fn: Function to time.
        *args, **kwargs: Passed to fn.

    Returns:
        TimingStats with per-iteration millisecond timings.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    return compute_stats(samples)


def cpu_timer(warmup: int, iters: int, fn: callable, *args, **kwargs) -> TimingStats:
    """Time a CPU function using time.perf_counter.

    Args:
        warmup: Number of warmup iterations (not measured).
        iters: Number of measured iterations.
        fn: Function to time.
        *args, **kwargs: Passed to fn.

    Returns:
        TimingStats with per-iteration millisecond timings.
    """
    for _ in range(warmup):
        fn(*args, **kwargs)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    return compute_stats(samples)


@contextmanager
def cuda_sync_timer() -> Generator[list[float], None, None]:
    """Context manager that yields a list; on exit, appends elapsed ms.

    Usage:
        with cuda_sync_timer() as t:
            do_gpu_work()
        elapsed_ms = t[0]
    """
    torch.cuda.synchronize()
    result: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield result
    finally:
        end.record()
        torch.cuda.synchronize()
        result.append(start.elapsed_time(end))
