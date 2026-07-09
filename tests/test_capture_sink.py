"""CaptureSink seam: throughput accounting with the fork-free in-memory sink.

No vllm / fork import: exercises the engine-neutral sink + recorder math.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from steering_bench.engine.capture_sink import (
    CaptureSinkError,
    InMemoryCaptureSink,
    SinkConfig,
    ThroughputRecorder,
    WriteChunk,
    WriteFinalize,
    make_capture_sink,
)


def _chunk(req: int, payload: bytes, append: bool = False) -> WriteChunk:
    return WriteChunk(
        key=(f"req_{req}", 0, "post_block"),
        path=Path(f"/tmp/{req}.bin"),
        payload=payload,
        append=append,
    )


def _finalize(req: int) -> WriteFinalize:
    return WriteFinalize(
        key=(f"req_{req}", 0, "post_block"),
        bin_path=Path(f"/tmp/{req}.bin"),
        sidecar_path=Path(f"/tmp/{req}.json"),
        sidecar_payload={"req_id": f"req_{req}"},
    )


def test_in_memory_sink_accounts_bytes_and_completes() -> None:
    sink = InMemoryCaptureSink()
    sink.start()

    n_req = 4
    steps = 8
    payload = b"x" * 1024  # 1 KiB per chunk
    for req in range(n_req):
        for step in range(steps):
            sink.submit_chunk(_chunk(req, payload, append=step > 0))
        sink.submit_finalize(_finalize(req))

    assert sink.wait_for_all(n_req, timeout=1.0) == n_req
    report = sink.report()

    total_chunks = n_req * steps
    expected_mb = (total_chunks * len(payload)) / (1024 * 1024)
    assert report.completed == n_req
    assert report.total_mb == pytest.approx(expected_mb, rel=1e-9)
    # Positive, finite throughput; chunks/s consistent with the elapsed window.
    assert report.throughput_mb_s > 0
    assert report.chunks_per_s == pytest.approx(
        total_chunks / report.total_seconds, rel=1e-9
    )
    # Synchronous finalize -> percentiles are present and non-negative.
    assert report.finalize_p50_ms is not None
    assert report.finalize_p99_ms is not None
    assert report.finalize_p50_ms >= 0


def test_wait_for_result_per_key() -> None:
    sink = InMemoryCaptureSink()
    sink.start()
    sink.submit_chunk(_chunk(0, b"abcd"))
    assert sink.wait_for_result(("req_0", 0, "post_block"), timeout=0.1) is False
    sink.submit_finalize(_finalize(0))
    assert sink.wait_for_result(("req_0", 0, "post_block"), timeout=0.1) is True


def test_throughput_mb_s_math_is_exact() -> None:
    """MB/s = total_MB / total_seconds; drive the recorder with a known window."""
    rec = ThroughputRecorder()
    rec.start()
    # 2 MiB across 2 chunks.
    rec.record_chunk(1024 * 1024)
    rec.record_chunk(1024 * 1024)
    time.sleep(0.02)
    rec.stop()
    report = rec.report()
    assert report.total_mb == pytest.approx(2.0, rel=1e-9)
    assert report.throughput_mb_s == pytest.approx(
        report.total_mb / report.total_seconds, rel=1e-9
    )
    # No finalizes recorded -> no percentiles.
    assert report.finalize_p50_ms is None
    assert report.finalize_p99_ms is None


def test_finalize_percentiles_ordered() -> None:
    rec = ThroughputRecorder()
    rec.start()
    # Two finalizes with deliberately different latencies.
    for i, delay in enumerate((0.0, 0.01)):
        key = (f"req_{i}", 0, "h")
        rec.mark_finalize_submit(key)
        time.sleep(delay)
        rec.mark_finalize_done(key)
    rec.stop()
    report = rec.report()
    assert report.completed == 2
    assert report.finalize_p99_ms >= report.finalize_p50_ms >= 0


def test_make_capture_sink_memory() -> None:
    sink = make_capture_sink("memory")
    assert isinstance(sink, InMemoryCaptureSink)


def test_make_capture_sink_vllm_requires_config() -> None:
    with pytest.raises(CaptureSinkError):
        make_capture_sink("vllm", None)


def test_make_capture_sink_unknown() -> None:
    with pytest.raises(CaptureSinkError):
        make_capture_sink("nope")


def test_sink_config_defaults() -> None:
    cfg = SinkConfig(root=Path("/tmp/x"))
    assert cfg.num_threads == 4
    assert cfg.fsync is True
    assert cfg.atomic_publish is True
