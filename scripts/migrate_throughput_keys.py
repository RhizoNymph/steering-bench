#!/usr/bin/env python3
"""One-time migration: rename throughput stat keys from *_ms to *_tps.

Older bench_throughput.py runs wrote the throughput_tokens_per_sec stats
with _ms suffixes (because TimingStats.to_dict() hardcodes those names),
even though the unit was tokens/sec. This script rewrites those files
in place with the corrected key names.

Idempotent: files already migrated are skipped. Safe: writes to a temp
file and atomically renames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keys in the throughput_tokens_per_sec dict that need renaming.
MS_TO_TPS_KEYS = {
    "mean_ms": "mean_tps",
    "median_ms": "median_tps",
    "stddev_ms": "stddev_tps",
    "p10_ms": "p10_tps",
    "p25_ms": "p25_tps",
    "p50_ms": "p50_tps",
    "p75_ms": "p75_tps",
    "p90_ms": "p90_tps",
    "p99_ms": "p99_tps",
}


def migrate_file(path: Path, dry_run: bool = False) -> str:
    """Migrate a single throughput JSON file.

    Returns one of: "migrated", "already_migrated", "skipped_no_field",
    "skipped_not_throughput", "error".
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ERROR {path}: {e}")
        return "error"

    if data.get("benchmark") != "vllm.throughput":
        return "skipped_not_throughput"

    results = data.get("results", {})
    tps_dict = results.get("throughput_tokens_per_sec")
    if not isinstance(tps_dict, dict):
        return "skipped_no_field"

    # Check if any old keys are present
    old_keys_present = [k for k in MS_TO_TPS_KEYS if k in tps_dict]
    if not old_keys_present:
        return "already_migrated"

    # Rename keys
    new_tps_dict = {
        MS_TO_TPS_KEYS.get(k, k): v for k, v in tps_dict.items()
    }
    data["results"]["throughput_tokens_per_sec"] = new_tps_dict

    if dry_run:
        return "migrated"

    # Atomic write: write to temp file in same dir, then rename
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except OSError as e:
        print(f"  ERROR writing {path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return "error"

    return "migrated"


def main():
    parser = argparse.ArgumentParser(
        description="Rename throughput stat keys from *_ms to *_tps"
    )
    parser.add_argument(
        "--results-dir",
        default="results/",
        help="Directory to scan recursively for JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return

    counts = {
        "migrated": 0,
        "already_migrated": 0,
        "skipped_not_throughput": 0,
        "skipped_no_field": 0,
        "error": 0,
    }

    print(f"Scanning {results_dir} for JSON files...")
    files = sorted(results_dir.rglob("*.json"))
    print(f"Found {len(files)} JSON files\n")

    for path in files:
        status = migrate_file(path, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status == "migrated":
            prefix = "[DRY-RUN] would migrate" if args.dry_run else "migrated"
            print(f"  {prefix}: {path.relative_to(results_dir)}")

    print(f"\n{'=' * 60}")
    print("  Migration Summary")
    print(f"{'=' * 60}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")
    if args.dry_run:
        print("DRY RUN — no files were modified. Rerun without --dry-run to apply.")


if __name__ == "__main__":
    main()
