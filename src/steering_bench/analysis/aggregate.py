"""Aggregate benchmark JSON results into a pandas DataFrame.

Reads all results/**/*.json, validates the schema, flattens
nested parameters and results into columns, and computes
derived metrics like overhead_pct and speedup_vs_baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"benchmark", "timestamp", "parameters", "results"}


def load_results(results_dir: str | Path) -> list[dict]:
    """Walk results_dir recursively, parse all JSON files.

    Skips files that fail validation with a warning.
    """
    results_dir = Path(results_dir)
    records = []

    if not results_dir.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        return records

    for path in sorted(results_dir.rglob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", path, e)
            continue

        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            logger.warning("Skipping %s: missing keys %s", path, missing)
            continue

        data["_source_file"] = str(path)
        records.append(data)

    logger.info("Loaded %d result files from %s", len(records), results_dir)
    return records


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict with underscore-separated keys."""
    items: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
            # Skip raw sample arrays — too large for DataFrame columns
            if "samples" in k.lower():
                continue
            items[key] = v
        else:
            items[key] = v
    return items


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten records into a DataFrame.

    Each record's parameters.* and results.* are flattened into columns
    with prefix-based naming (e.g. results_latency_ms_mean_ms).
    """
    rows = []
    for rec in records:
        row: dict = {
            "benchmark": rec["benchmark"],
            "timestamp": rec["timestamp"],
            "tag": rec.get("tag", ""),
            "_source_file": rec.get("_source_file", ""),
        }

        # Environment metadata
        env = rec.get("environment", {})
        row["env_gpu"] = env.get("gpu", "unknown")
        row["env_hostname"] = env.get("hostname", "unknown")
        row["env_vllm_version"] = env.get("vllm_version", "unknown")

        # Flatten parameters
        params = rec.get("parameters", {})
        row.update(_flatten_dict(params, "param"))

        # Flatten results
        results = rec.get("results", {})
        row.update(_flatten_dict(results, "result"))

        rows.append(row)

    return pd.DataFrame(rows)


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: overhead_pct, speedup_vs_baseline."""
    df = df.copy()

    # --- vLLM latency overhead ---
    latency_mask = df["benchmark"] == "vllm.latency"
    if latency_mask.any():
        latency_df = df[latency_mask].copy()
        # Build per-(tag, batch_size) baselines from the disabled-mode rows so
        # multiple tagged runs in the same results dir (e.g. pre-fix vs
        # post-fix) compute overhead against their own same-tag baseline
        # instead of colliding in a single index.
        tag_col = "tag" if "tag" in latency_df.columns else None
        disabled_df = latency_df[latency_df["param_mode"] == "disabled"]
        if tag_col is not None:
            baseline_group = disabled_df.groupby(
                [tag_col, "param_batch_size"]
            )["result_latency_ms_mean_ms"].mean()
        else:
            baseline_group = disabled_df.groupby(
                "param_batch_size"
            )["result_latency_ms_mean_ms"].mean()

        def calc_overhead(row):
            bs = row.get("param_batch_size")
            mean = row.get("result_latency_ms_mean_ms")
            if pd.isna(bs) or pd.isna(mean):
                return None
            if tag_col is not None:
                key = (row.get(tag_col, ""), bs)
            else:
                key = bs
            if key not in baseline_group.index:
                return None
            base = baseline_group[key]
            if base <= 0:
                return None
            return (mean - base) / base * 100

        df.loc[latency_mask, "derived_overhead_pct"] = latency_df.apply(
            calc_overhead, axis=1
        )

    # --- External library speedup vs HF baseline ---
    tier1_mask = df["benchmark"].str.startswith("external.tier1.")
    if tier1_mask.any():
        tier1_df = df[tier1_mask].copy()
        baseline_row = tier1_df[tier1_df["benchmark"] == "external.tier1.hf_baseline"]
        if not baseline_row.empty:
            baseline_mean = baseline_row.iloc[0].get("result_latency_ms_mean_ms")
            if baseline_mean and baseline_mean > 0:
                df.loc[tier1_mask, "derived_speedup_vs_baseline"] = (
                    baseline_mean / df.loc[tier1_mask, "result_latency_ms_mean_ms"]
                )

    # --- Memory overhead ---
    memory_mask = df["benchmark"] == "vllm.memory"
    if memory_mask.any():
        memory_df = df[memory_mask].copy()
        baseline_mem = memory_df[memory_df["param_max_steering_configs"] == 0]
        if not baseline_mem.empty:
            base_alloc = baseline_mem.iloc[0].get("result_allocated_mb")
            if base_alloc and base_alloc > 0:
                df.loc[memory_mask, "derived_memory_overhead_mb"] = (
                    df.loc[memory_mask, "result_allocated_mb"] - base_alloc
                )

    return df


def aggregate(results_dir: str | Path) -> pd.DataFrame:
    """Load, flatten, and compute derived metrics for all results.

    Convenience function combining load_results, to_dataframe,
    and compute_derived.
    """
    records = load_results(results_dir)
    if not records:
        return pd.DataFrame()
    df = to_dataframe(records)
    df = compute_derived(df)
    return df
