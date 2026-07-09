"""TTFT/TPOT/ITL/E2EL from synthetic token timestamps. No live server."""

from __future__ import annotations

import pytest

from steering_bench.engine.serving import (
    RequestResult,
    compute_request_metrics,
    summarize_results,
)


def test_metrics_from_known_deltas() -> None:
    t0 = 100.0
    # first token at +0.010s, then +0.005s gaps.
    token_times = [100.010, 100.015, 100.020, 100.025]
    end = 100.030
    r = compute_request_metrics(t0, token_times, end_time=end)
    assert r.num_output_tokens == 4
    assert r.ttft_ms == pytest.approx(10.0)
    assert r.itl_ms == pytest.approx([5.0, 5.0, 5.0])
    assert r.e2el_ms == pytest.approx(30.0)
    assert r.error is None


def test_metrics_default_end_time_is_last_token() -> None:
    r = compute_request_metrics(0.0, [0.01, 0.02])
    assert r.ttft_ms == pytest.approx(10.0)
    assert r.e2el_ms == pytest.approx(20.0)
    assert r.itl_ms == pytest.approx([10.0])


def test_metrics_single_token_no_itl() -> None:
    r = compute_request_metrics(0.0, [0.007], end_time=0.009)
    assert r.num_output_tokens == 1
    assert r.ttft_ms == pytest.approx(7.0)
    assert r.itl_ms == []
    assert r.e2el_ms == pytest.approx(9.0)


def test_metrics_empty_stream() -> None:
    r = compute_request_metrics(0.0, [], end_time=0.005)
    assert r.num_output_tokens == 0
    assert r.ttft_ms is None
    assert r.e2el_ms == pytest.approx(5.0)


def test_summarize_tpot_and_families() -> None:
    # Two identical requests: ttft=10, 4 tokens, e2el=30.
    # TPOT = (30 - 10) / (4 - 1) = 6.667 ms/token.
    results = [
        compute_request_metrics(0.0, [0.010, 0.015, 0.020, 0.025], end_time=0.030)
        for _ in range(2)
    ]
    s = summarize_results(results)
    assert s["num_ok"] == 2
    assert s["num_err"] == 0
    assert s["ttft_ms"]["median_ms"] == pytest.approx(10.0)
    assert s["tpot_ms"]["median_ms"] == pytest.approx(20.0 / 3.0)
    assert s["itl_ms"]["median_ms"] == pytest.approx(5.0)
    assert s["e2el_ms"]["median_ms"] == pytest.approx(30.0)
    assert s["total_output_tokens"] == 8
    # offline_output_tps = total_out / (max_e2el_s). max e2el = 0.030 s.
    assert abs(s["offline_output_tps"] - 8 / 0.030) < 1e-6


def test_summarize_all_failed() -> None:
    s = summarize_results([RequestResult(error="boom") for _ in range(3)])
    assert s["num_ok"] == 0
    assert s["num_err"] == 3
    assert s["errors"] == ["boom", "boom", "boom"]


def test_summarize_mixed_ok_and_err() -> None:
    ok = compute_request_metrics(0.0, [0.01, 0.02], end_time=0.03)
    err = RequestResult(error="TypeError: x")
    s = summarize_results([ok, err])
    assert s["num_ok"] == 1
    assert s["num_err"] == 1
    assert s["errors"] == ["TypeError: x"]
