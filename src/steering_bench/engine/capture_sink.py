"""CaptureSink seam: an engine-neutral activation-writer / throughput surface.

The filesystem writer benchmark (``bench_capture_filesystem.py``) drives the
fork's ``ActivationWriter`` directly -- no LLM, no capture manager -- to answer
"can the writer keep up with the model?". This module generalizes that writer so
the benchmark is engine-neutral: it constructs a :class:`CaptureSink` through the
seam and feeds it typed :class:`WriteChunk` / :class:`WriteFinalize` payloads,
and the sink reports :class:`SinkThroughput` (MB/s, chunks/s, finalize p50/p99).

Two implementations:
  * :class:`InMemoryCaptureSink` -- discards payloads, finalizes synchronously,
    accounts throughput from the byte counts fed to it. No fork import; used by
    tests and as a "no I/O" reference.
  * :class:`VllmActivationWriterSink` -- wraps ``ActivationWriter`` (imported
    LAZILY), mapping ``WriteChunk`` -> ``WriteTask`` and ``WriteFinalize`` ->
    ``FinalizeTask`` and blocking on the writer's status callback.

The ``consumers.py`` sinks (``NullCaptureSink`` etc.) already implement the
fork's ``CaptureChunk``/``CaptureFinalize`` protocol at a different level (the
manager's dispatch path); they are unchanged. This seam is the writer-throughput
level the filesystem benchmark measures.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# (req_id, layer, hook_name) -- the writer's per-capture key.
CaptureKey = tuple[str, int, str]


class CaptureSinkError(RuntimeError):
    """Raised when a capture sink is misused or a backend is unavailable."""


@dataclass(frozen=True)
class WriteChunk:
    """One activation write: ``payload`` bytes appended/created at ``path``."""

    key: CaptureKey
    path: Path
    payload: bytes
    append: bool = False


@dataclass(frozen=True)
class WriteFinalize:
    """Finalize one capture: publish ``bin_path`` + write ``sidecar_path``."""

    key: CaptureKey
    bin_path: Path
    sidecar_path: Path
    sidecar_payload: dict[str, Any]


@dataclass(frozen=True)
class SinkThroughput:
    """Throughput / latency report for a sink run."""

    total_mb: float
    total_seconds: float
    throughput_mb_s: float
    chunks_per_s: float
    completed: int
    finalize_p50_ms: float | None
    finalize_p99_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mb": self.total_mb,
            "total_seconds": self.total_seconds,
            "throughput_mb_s": self.throughput_mb_s,
            "chunks_per_s": self.chunks_per_s,
            "completed": self.completed,
            "finalize_p50_ms": self.finalize_p50_ms,
            "finalize_p99_ms": self.finalize_p99_ms,
        }


class ThroughputRecorder:
    """Bytes / chunk-count / finalize-latency accounting for a sink run.

    Engine-neutral: both sinks embed one. ``start`` stamps the run origin,
    ``record_chunk`` accumulates written bytes, ``mark_finalize_submit`` /
    ``mark_finalize_done`` bracket each finalize for its latency, and ``report``
    computes MB/s, chunks/s and finalize p50/p99. Thread-safe: the writer's
    status callback fires on a background thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._total_bytes = 0
        self._chunks = 0
        self._submit_times: dict[CaptureKey, float] = {}
        self._done_times: dict[CaptureKey, float] = {}

    def start(self) -> None:
        with self._lock:
            self._t_start = time.perf_counter()

    def stop(self) -> None:
        with self._lock:
            self._t_end = time.perf_counter()

    def record_chunk(self, nbytes: int) -> None:
        with self._lock:
            self._total_bytes += int(nbytes)
            self._chunks += 1

    def mark_finalize_submit(self, key: CaptureKey) -> None:
        with self._lock:
            self._submit_times[key] = time.perf_counter()

    def mark_finalize_done(self, key: CaptureKey) -> None:
        with self._lock:
            self._done_times[key] = time.perf_counter()

    @property
    def completed(self) -> int:
        with self._lock:
            return len(self._done_times)

    def report(self) -> SinkThroughput:
        with self._lock:
            t_start = self._t_start if self._t_start is not None else time.perf_counter()
            t_end = self._t_end if self._t_end is not None else time.perf_counter()
            total_seconds = max(t_end - t_start, 1e-12)
            total_mb = self._total_bytes / (1024 * 1024)
            throughput = total_mb / total_seconds
            chunks_per_s = self._chunks / total_seconds
            latencies_ms = [
                (self._done_times[k] - self._submit_times[k]) * 1000.0
                for k in self._done_times
                if k in self._submit_times
            ]
            completed = len(self._done_times)
        p50 = p99 = None
        if latencies_ms:
            arr = np.array(latencies_ms)
            p50 = float(np.percentile(arr, 50))
            p99 = float(np.percentile(arr, 99))
        return SinkThroughput(
            total_mb=total_mb,
            total_seconds=total_seconds,
            throughput_mb_s=throughput,
            chunks_per_s=chunks_per_s,
            completed=completed,
            finalize_p50_ms=p50,
            finalize_p99_ms=p99,
        )


class CaptureSink(abc.ABC):
    """Engine-neutral activation-writer sink.

    Submit ``WriteChunk`` / ``WriteFinalize`` payloads, wait for finalizes, and
    read a :class:`SinkThroughput` report. The benchmark constructs one via
    :func:`make_capture_sink` and never touches a fork type directly.
    """

    def __init__(self) -> None:
        self._recorder = ThroughputRecorder()

    def start(self) -> None:
        """Stamp the run origin (call just before submitting the first chunk)."""
        self._recorder.start()

    @abc.abstractmethod
    def submit_chunk(self, chunk: WriteChunk) -> None:
        """Submit one activation write."""

    @abc.abstractmethod
    def submit_finalize(self, finalize: WriteFinalize) -> None:
        """Submit one finalize (publish bin + sidecar)."""

    @abc.abstractmethod
    def wait_for_result(self, key: CaptureKey, timeout: float) -> bool:
        """Block until ``key`` finalizes (or ``timeout``); True if it completed."""

    @abc.abstractmethod
    def wait_for_all(self, expected: int, timeout: float) -> int:
        """Block until ``expected`` finalizes complete (or ``timeout``).

        Returns the number actually completed.
        """

    @abc.abstractmethod
    def shutdown(self, timeout: float = 30.0) -> None:
        """Flush and stop the sink."""

    def report(self) -> SinkThroughput:
        """Throughput / latency for the run so far."""
        self._recorder.stop()
        return self._recorder.report()


class InMemoryCaptureSink(CaptureSink):
    """Sink that discards payloads and finalizes synchronously.

    Accounts throughput from the byte counts fed to it -- a "no I/O" reference
    and the fake used by tests. No fork import.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._done: set[CaptureKey] = set()

    def submit_chunk(self, chunk: WriteChunk) -> None:
        self._recorder.record_chunk(len(chunk.payload))

    def submit_finalize(self, finalize: WriteFinalize) -> None:
        self._recorder.mark_finalize_submit(finalize.key)
        with self._lock:
            self._done.add(finalize.key)
        self._recorder.mark_finalize_done(finalize.key)

    def wait_for_result(self, key: CaptureKey, timeout: float) -> bool:
        with self._lock:
            return key in self._done

    def wait_for_all(self, expected: int, timeout: float) -> int:
        return self._recorder.completed

    def shutdown(self, timeout: float = 30.0) -> None:
        pass


class VllmActivationWriterSink(CaptureSink):
    """Sink wrapping the fork's ``ActivationWriter`` (imported lazily).

    Maps ``WriteChunk`` -> ``WriteTask`` and ``WriteFinalize`` -> ``FinalizeTask``
    and installs a status callback so ``wait_for_*`` blocks on completion and the
    recorder times each finalize.
    """

    def __init__(
        self,
        root: Path,
        *,
        num_threads: int = 4,
        queue_size: int = 4096,
        on_collision: str = "overwrite",
        fsync: bool = True,
        atomic_publish: bool = True,
    ) -> None:
        super().__init__()
        from vllm.v1.capture.consumers.filesystem.writer import ActivationWriter

        self._writer = ActivationWriter(
            root,
            num_threads=num_threads,
            queue_size=queue_size,
            on_collision=on_collision,
            fsync=fsync,
            atomic_publish=atomic_publish,
        )
        self._lock = threading.Lock()
        self._done: set[CaptureKey] = set()
        self._cond = threading.Condition(self._lock)
        self._writer.add_status_callback(self._on_status)

    def _on_status(self, result: Any) -> None:
        if result.status in ("ok", "error"):
            key = result.key
            self._recorder.mark_finalize_done(key)
            with self._cond:
                self._done.add(key)
                self._cond.notify_all()

    def submit_chunk(self, chunk: WriteChunk) -> None:
        from vllm.v1.capture.consumers.filesystem.writer import WriteTask

        self._recorder.record_chunk(len(chunk.payload))
        self._writer.submit(
            WriteTask(
                path=chunk.path,
                payload=chunk.payload,
                append=chunk.append,
                key=chunk.key,
            )
        )

    def submit_finalize(self, finalize: WriteFinalize) -> None:
        from vllm.v1.capture.consumers.filesystem.writer import FinalizeTask

        self._recorder.mark_finalize_submit(finalize.key)
        self._writer.submit(
            FinalizeTask(
                bin_path=finalize.bin_path,
                sidecar_path=finalize.sidecar_path,
                sidecar_payload=finalize.sidecar_payload,
                key=finalize.key,
            )
        )

    def wait_for_result(self, key: CaptureKey, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        with self._cond:
            while key not in self._done:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def wait_for_all(self, expected: int, timeout: float) -> int:
        deadline = time.perf_counter() + timeout
        with self._cond:
            while len(self._done) < expected:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return len(self._done)

    def shutdown(self, timeout: float = 30.0) -> None:
        self._writer.shutdown(timeout=timeout)


@dataclass(frozen=True)
class SinkConfig:
    """Construction knobs for a filesystem-backed :class:`CaptureSink`."""

    root: Path
    num_threads: int = 4
    queue_size: int = 4096
    on_collision: str = "overwrite"
    fsync: bool = True
    atomic_publish: bool = True


def make_capture_sink(engine: str, config: SinkConfig | None = None) -> CaptureSink:
    """Construct a :class:`CaptureSink` for ``engine`` through the seam.

    ``"vllm"`` -> :class:`VllmActivationWriterSink` (needs a ``config`` with a
    writable ``root``); ``"memory"`` -> :class:`InMemoryCaptureSink` (no I/O).
    """
    match engine:
        case "vllm":
            if config is None:
                raise CaptureSinkError("vllm capture sink requires a SinkConfig root")
            return VllmActivationWriterSink(
                config.root,
                num_threads=config.num_threads,
                queue_size=config.queue_size,
                on_collision=config.on_collision,
                fsync=config.fsync,
                atomic_publish=config.atomic_publish,
            )
        case "memory":
            return InMemoryCaptureSink()
    raise CaptureSinkError(f"unknown capture sink engine {engine!r}")
