#!/usr/bin/env python3
"""Compare throughput benchmark runs across tags.

Reads all vllm.throughput JSON files, filters by tag, and prints
a comparison table showing tokens/sec and throughput loss for
each experimental condition.

Example:
    python scripts/compare_throughput.py \\
        --tags baseline-v2,no-prefix-cache,big-table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_throughput_results(results_dir: Path, tags: list[str] | None) -> list[dict]:
    """Load all vllm.throughput JSON records, optionally filtered by tag."""
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: skipping {path}: {e}", file=sys.stderr)
            continue
        if data.get("benchmark") != "vllm.throughput":
            continue
        if tags and data.get("tag", "") not in tags:
            continue
        records.append(data)
    return records


def get_tps(rec: dict) -> float | None:
    """Extract mean tokens/sec from a record, handling both old and new key names."""
    tps_dict = rec.get("results", {}).get("throughput_tokens_per_sec", {})
    if not isinstance(tps_dict, dict):
        return None
    # New schema uses mean_tps; old schema used mean_ms
    return tps_dict.get("mean_tps") or tps_dict.get("mean_ms")


def get_latency_ms(rec: dict) -> float | None:
    lat = rec.get("results", {}).get("latency_ms", {})
    if not isinstance(lat, dict):
        return None
    return lat.get("mean_ms")


def format_cell(value: float | None, width: int = 10, fmt: str = ".0f") -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:>{width}{fmt}}"


def main():
    parser = argparse.ArgumentParser(description="Compare throughput benchmark runs")
    parser.add_argument("--results-dir", default="results/vllm/")
    parser.add_argument(
        "--tags",
        default="baseline-v2,no-prefix-cache,big-table",
        help="Comma-separated tags to compare (order = column order)",
    )
    args = parser.parse_args()

    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    results_dir = Path(args.results_dir)

    records = load_throughput_results(results_dir, tag_list)
    if not records:
        print(f"No vllm.throughput records found in {results_dir} for tags: {tag_list}")
        sys.exit(1)

    print(f"Loaded {len(records)} throughput records")
    print(f"Tags: {tag_list}")
    print()

    # Index: (tag, distinct_configs) -> record
    by_key: dict[tuple[str, int], dict] = {}
    for rec in records:
        tag = rec.get("tag", "")
        distinct = rec.get("parameters", {}).get("distinct_configs")
        if distinct is None:
            continue
        by_key[(tag, distinct)] = rec

    all_distincts = sorted(set(d for _, d in by_key.keys()))

    # --- Throughput comparison table ---
    print(f"{'=' * 100}")
    print("  Throughput (tokens/sec) by distinct_configs x tag")
    print(f"{'=' * 100}")
    header = f"{'distinct':>10}"
    for tag in tag_list:
        header += f" {tag:>20}"
    print(header)
    print(f"{'-' * 100}")

    # Row for each distinct_configs value
    for distinct in all_distincts:
        row = f"{distinct:>10}"
        for tag in tag_list:
            rec = by_key.get((tag, distinct))
            tps = get_tps(rec) if rec else None
            row += f" {format_cell(tps, width=20)}"
        print(row)
    print()

    # --- Throughput loss vs tag's own configs=0 baseline ---
    print(f"{'=' * 100}")
    print("  Throughput loss (%) vs same-tag configs=0 baseline")
    print(f"{'=' * 100}")
    header = f"{'distinct':>10}"
    for tag in tag_list:
        header += f" {tag:>20}"
    print(header)
    print(f"{'-' * 100}")

    # Compute baseline per tag
    baselines: dict[str, float] = {}
    for tag in tag_list:
        base_rec = by_key.get((tag, 0))
        if base_rec:
            tps = get_tps(base_rec)
            if tps:
                baselines[tag] = tps

    for distinct in all_distincts:
        row = f"{distinct:>10}"
        for tag in tag_list:
            rec = by_key.get((tag, distinct))
            tps = get_tps(rec) if rec else None
            baseline = baselines.get(tag)
            if tps is None or baseline is None or baseline <= 0:
                row += f" {'—':>20}"
            else:
                loss_pct = (baseline - tps) / baseline * 100
                row += f" {loss_pct:>19.1f}%"
        print(row)
    print()

    # --- Cross-tag comparisons: how does each tag compare to baseline-v2? ---
    if "baseline-v2" in tag_list and len(tag_list) > 1:
        print(f"{'=' * 100}")
        print("  Speedup vs baseline-v2 (tokens/sec ratio)")
        print(f"{'=' * 100}")
        other_tags = [t for t in tag_list if t != "baseline-v2"]
        header = f"{'distinct':>10} {'baseline-v2':>16}"
        for tag in other_tags:
            header += f" {tag:>20}"
        print(header)
        print(f"{'-' * 100}")

        for distinct in all_distincts:
            baseline_rec = by_key.get(("baseline-v2", distinct))
            baseline_tps = get_tps(baseline_rec) if baseline_rec else None
            row = f"{distinct:>10} {format_cell(baseline_tps, width=16)}"
            for tag in other_tags:
                rec = by_key.get((tag, distinct))
                tps = get_tps(rec) if rec else None
                if tps is None or baseline_tps is None or baseline_tps <= 0:
                    row += f" {'—':>20}"
                else:
                    ratio = tps / baseline_tps
                    row += f" {ratio:>17.2f}x"
            print(row)
        print()

    # --- Parameters summary for each tag ---
    print(f"{'=' * 100}")
    print("  Parameters by tag")
    print(f"{'=' * 100}")
    for tag in tag_list:
        # Find any record with this tag
        sample = next((r for r in records if r.get("tag") == tag), None)
        if not sample:
            print(f"  {tag}: NO RECORDS")
            continue
        params = sample.get("parameters", {})
        prefix_cache = params.get("prefix_caching", "?")
        max_configs = params.get("max_steering_configs", "?")
        num_prompts = params.get("num_prompts", "?")
        prompt_len = params.get("prompt_len", "?")
        max_tokens = params.get("max_tokens", "?")
        print(
            f"  {tag}: prefix_cache={prefix_cache}, max_steering_configs={max_configs}, "
            f"num_prompts={num_prompts}, prompt_len={prompt_len}, max_tokens={max_tokens}"
        )


if __name__ == "__main__":
    main()
