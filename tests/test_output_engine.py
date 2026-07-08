"""First-class engine identity in the result schema (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path

from steering_bench.analysis.aggregate import to_dataframe
from steering_bench.output import write_result


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_engine_block_round_trips(tmp_path: Path) -> None:
    engine = {"name": "vllm", "version": "0.11.0", "commit": "abc123"}
    path = write_result(
        benchmark="unit.engine",
        parameters={"model": "fake/model"},
        results={"latency_ms": {"mean_ms": 1.0}},
        output_dir=tmp_path,
        engine=engine,
    )
    data = _read(path)
    assert data["engine"] == engine


def test_engine_block_absent_when_not_provided(tmp_path: Path) -> None:
    path = write_result(
        benchmark="unit.no_engine",
        parameters={"model": "fake/model"},
        results={"latency_ms": {"mean_ms": 1.0}},
        output_dir=tmp_path,
    )
    assert "engine" not in _read(path)


def test_aggregate_produces_engine_columns(tmp_path: Path) -> None:
    write_result(
        benchmark="unit.engine",
        parameters={"model": "fake/model"},
        results={"latency_ms": {"mean_ms": 1.0}},
        output_dir=tmp_path,
        engine={"name": "transformerlens", "version": "2.1", "commit": None},
    )
    records = [_read(p) for p in sorted(tmp_path.glob("*.json"))]
    df = to_dataframe(records)
    assert "engine_name" in df.columns
    assert "engine_version" in df.columns
    assert "engine_commit" in df.columns
    assert df.iloc[0]["engine_name"] == "transformerlens"
    assert df.iloc[0]["engine_version"] == "2.1"


def test_aggregate_tolerates_records_without_engine_block(tmp_path: Path) -> None:
    # A record shaped like the legacy schema (no engine block).
    legacy = {
        "benchmark": "vllm.latency",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "tag": "",
        "environment": {"gpu": "none", "vllm_version": "0.10"},
        "parameters": {"model": "m", "mode": "disabled"},
        "results": {"latency_ms": {"mean_ms": 2.0}},
    }
    df = to_dataframe([legacy])
    assert len(df) == 1
    # Column exists but the value is missing (None) for legacy records.
    assert "engine_name" in df.columns
    assert df.iloc[0]["engine_name"] is None
