"""Reusable consumer helpers for capture benchmarks.

``NullCaptureSink`` discards all data immediately so that
``bench_capture_manager.py`` can measure manager overhead without any
I/O confounding.

``RecordingDriverConsumer`` is a driver-side consumer that accumulates
captures in memory; used by ``bench_capture_e2e.py`` to verify the
pipeline is live while measuring end-to-end overhead.
"""

from __future__ import annotations

import threading
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
