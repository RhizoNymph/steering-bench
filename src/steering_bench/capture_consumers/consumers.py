"""Reusable consumer helpers for capture benchmarks.

``NullCaptureSink`` discards all data immediately so that
``bench_capture_manager.py`` can measure manager overhead without any
I/O confounding.

``RecordingDriverConsumer`` is a driver-side consumer that accumulates
captures in memory; used by ``bench_capture_e2e.py`` to verify the
pipeline is live while measuring end-to-end overhead.

``TimestampingSink`` / ``TimestampingConsumer`` record perf_counter_ns
stamps at chunk delivery / on_capture firing for the latency benchmark.

``SimulatedWorkSink`` / ``SimulatedWorkConsumer`` spend a configurable
number of microseconds per chunk (busy-wait, sleep, or bounded queue)
for the plugin-work benchmark.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, ClassVar, Literal

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)


class NullCaptureSink:
    """CaptureSink that discards all submitted data.

    Fulfills the CaptureSink protocol without any I/O so that
    bench_capture_manager.py can isolate manager CPU and GPU overhead.
    Results are immediately marked "ok" on finalize.
    """

    location: ClassVar[Literal["worker"]] = "worker"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[CaptureKey, CaptureResult] = {}

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        pass

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        with self._lock:
            self._results[finalize.key] = CaptureResult(
                key=finalize.key, status="ok"
            )

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def wait_for_result(self, key: CaptureKey, timeout: float) -> CaptureResult | None:
        return self.get_result(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        pass

    def clear(self) -> None:
        with self._lock:
            self._results.clear()


class RecordingDriverConsumer(CaptureConsumer):
    """Driver consumer that counts captures for benchmark validation.

    Used by bench_capture_e2e.py as a minimal driver-side consumer that
    confirms the capture pipeline is live while measuring end-to-end
    throughput overhead. Intentionally stores no tensor data to avoid
    clone/allocation overhead inflating measurements.
    """

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(
        self,
        hooks: dict[str, list[int]],
        positions: str = "last_prompt",
    ) -> None:
        self._spec = CaptureSpec(hooks=hooks, positions=positions)
        self._count: int = 0
        self._lock = threading.Lock()

    def global_capture_spec(self) -> CaptureSpec:
        return self._spec

    def on_capture(
        self,
        key: CaptureKey,
        tensor: Any,
        sidecar: dict[str, Any],
    ) -> None:
        with self._lock:
            self._count += 1

    def clear(self) -> None:
        with self._lock:
            self._count = 0

    def count(self) -> int:
        with self._lock:
            return self._count


# ---------------------------------------------------------------------------
# Latency benchmark helpers
# ---------------------------------------------------------------------------


class TimestampingSink:
    """CaptureSink that records ``perf_counter_ns`` on each submit_chunk.

    Used by ``bench_capture_latency.py`` in microbench mode to measure
    dispatch-added delivery latency.  The benchmark brackets a
    ``dispatch_step_captures`` call with a ``t_dispatch_start`` stamp
    and then diffs against the per-chunk stamps recorded here.

    ``drain_timestamps`` returns and clears the buffered stamps; call it
    once per iteration so samples don't accumulate across warmup/measure
    boundaries.
    """

    location: ClassVar[Literal["worker"]] = "worker"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[CaptureKey, CaptureResult] = {}
        self._timestamps: list[tuple[CaptureKey, int, int]] = []

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        ts = time.perf_counter_ns()
        with self._lock:
            self._timestamps.append((chunk.key, chunk.step_index, ts))

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        with self._lock:
            self._results[finalize.key] = CaptureResult(
                key=finalize.key, status="ok"
            )

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def wait_for_result(self, key: CaptureKey, timeout: float) -> CaptureResult | None:
        return self.get_result(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        pass

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
            self._timestamps.clear()

    def drain_timestamps(self) -> list[tuple[CaptureKey, int, int]]:
        with self._lock:
            out = list(self._timestamps)
            self._timestamps.clear()
            return out


class TimestampingConsumer(CaptureConsumer):
    """CaptureConsumer that records ``perf_counter_ns`` in on_capture.

    Used by ``bench_capture_latency.py`` in E2E mode to measure the
    end-to-end delay from ``llm.generate()`` submission to the
    consumer's ``on_capture`` callback firing.  ``location`` is
    configurable per-instance (worker or driver) so the benchmark can
    sweep over it — driver-side exercises the worker→driver bridge
    (IPC queue + thread hop) while worker-side stays in-process.
    """

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(
        self,
        hooks: dict[str, list[int]],
        positions: str = "last_prompt",
        location: Literal["worker", "driver"] = "driver",
    ) -> None:
        # Instance-level shadow of the ClassVar so one class can serve
        # both worker and driver routing.  vLLM reads ``consumer.location``
        # via attribute lookup, which resolves the instance attr first.
        self.location = location  # type: ignore[misc]
        self._spec = CaptureSpec(hooks=hooks, positions=positions)
        self._lock = threading.Lock()
        self._timestamps: list[tuple[CaptureKey, int]] = []

    def global_capture_spec(self) -> CaptureSpec:
        return self._spec

    def on_capture(
        self,
        key: CaptureKey,
        tensor: Any,
        sidecar: dict[str, Any],
    ) -> None:
        ts = time.perf_counter_ns()
        with self._lock:
            self._timestamps.append((key, ts))

    def drain_timestamps(self) -> list[tuple[CaptureKey, int]]:
        with self._lock:
            out = list(self._timestamps)
            self._timestamps.clear()
            return out

    def clear(self) -> None:
        with self._lock:
            self._timestamps.clear()


# ---------------------------------------------------------------------------
# Plugin-work simulator
# ---------------------------------------------------------------------------


class _WorkSimulator:
    """Encapsulates "spend ``work_us`` per call" across three modes.

    - ``busy``: spin-wait on ``perf_counter_ns``.  Models a synchronous
      CPU-bound plugin.
    - ``sleep``: ``time.sleep(work_us / 1e6)``.  Models yielding work.
      Meaningless below ~100 μs due to scheduler resolution.
    - ``queue``: enqueue a sentinel; a single worker thread drains and
      sleeps per item.  Models a realistic queued consumer — submit_chunk
      returns immediately until the queue fills, at which point it
      blocks (backpressure).
    """

    def __init__(
        self,
        work_us: float,
        mode: Literal["busy", "sleep", "queue"],
        queue_depth: int = 64,
    ) -> None:
        self._work_us = float(work_us)
        self._mode = mode
        self._queue: queue.Queue[object] | None = None
        self._worker: threading.Thread | None = None
        if mode == "queue":
            self._queue = queue.Queue(maxsize=queue_depth)
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            if self._work_us > 0:
                time.sleep(self._work_us / 1e6)
            self._queue.task_done()

    def do_work(self) -> None:
        if self._mode == "busy":
            target_ns = int(self._work_us * 1000)
            if target_ns <= 0:
                return
            t0 = time.perf_counter_ns()
            while time.perf_counter_ns() - t0 < target_ns:
                pass
        elif self._mode == "sleep":
            if self._work_us > 0:
                time.sleep(self._work_us / 1e6)
        elif self._mode == "queue":
            assert self._queue is not None
            self._queue.put(1)
        else:
            msg = f"unknown work mode {self._mode!r}"
            raise ValueError(msg)

    def drain(self) -> None:
        """Wait for all queued work to complete (no-op except for queue mode)."""
        if self._queue is not None:
            self._queue.join()

    def shutdown(self) -> None:
        if self._queue is not None and self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5.0)


class SimulatedWorkSink:
    """CaptureSink that spends ``work_us`` per ``submit_chunk`` call.

    Used by ``bench_capture_plugin_work.py`` in microbench mode.  The
    manager's dispatch loop calls ``submit_chunk`` once per consumer per
    delivered chunk, so the sink's work cost shows up directly as
    additional ``dispatch_ms``.
    """

    location: ClassVar[Literal["worker"]] = "worker"

    def __init__(
        self,
        work_us: float,
        mode: Literal["busy", "sleep", "queue"] = "busy",
        queue_depth: int = 64,
    ) -> None:
        self._sim = _WorkSimulator(work_us, mode, queue_depth)
        self._lock = threading.Lock()
        self._results: dict[CaptureKey, CaptureResult] = {}

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        self._sim.do_work()

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        self._sim.drain()
        with self._lock:
            self._results[finalize.key] = CaptureResult(
                key=finalize.key, status="ok"
            )

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def wait_for_result(self, key: CaptureKey, timeout: float) -> CaptureResult | None:
        return self.get_result(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        self._sim.shutdown()

    def clear(self) -> None:
        with self._lock:
            self._results.clear()


class SimulatedWorkConsumer(CaptureConsumer):
    """CaptureConsumer that spends ``work_us`` per ``on_capture`` call.

    Used by ``bench_capture_plugin_work.py`` in E2E mode.  Driver-side
    by default — that's the path online reward consumers typically
    take, so throughput impact matches what users will see in practice.
    ``on_capture`` fires once per finalized capture key (not per chunk),
    so the work-per-call semantics differ from :class:`SimulatedWorkSink`;
    the benchmark reports both side-by-side.
    """

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(
        self,
        hooks: dict[str, list[int]],
        positions: str = "last_prompt",
        work_us: float = 0.0,
        mode: Literal["busy", "sleep", "queue"] = "busy",
        queue_depth: int = 64,
        location: Literal["worker", "driver"] = "driver",
    ) -> None:
        self.location = location  # type: ignore[misc]
        self._spec = CaptureSpec(hooks=hooks, positions=positions)
        self._sim = _WorkSimulator(work_us, mode, queue_depth)
        self._lock = threading.Lock()
        self._count = 0

    def global_capture_spec(self) -> CaptureSpec:
        return self._spec

    def on_capture(
        self,
        key: CaptureKey,
        tensor: Any,
        sidecar: dict[str, Any],
    ) -> None:
        self._sim.do_work()
        with self._lock:
            self._count += 1

    def count(self) -> int:
        with self._lock:
            return self._count

    def clear(self) -> None:
        self._sim.drain()
        with self._lock:
            self._count = 0

    def shutdown(self, timeout: float = 30.0) -> None:
        self._sim.shutdown()
