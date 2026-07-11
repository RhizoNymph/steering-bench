#!/usr/bin/env python3
"""Aggregate + diff torch.profiler chrome traces to isolate capture overhead.

Usage:
  analyze_trace.py LABEL=trace_or_dir [LABEL2=trace_or_dir2 ...]

Each arg is `label=path`; path is a .json/.json.gz chrome trace or a directory
containing one (the newest is used). Prints per-label GPU-kernel / memcpy /
cpu-op time grouped by op name, and — when exactly two labels are given —
diffs them (delta us and ops present only under capture).
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

# chrome-trace categories torch.profiler emits
GPU_KERNEL = {"kernel", "Kernel"}
MEMCPY = {"gpu_memcpy", "gpu_memset", "Memcpy", "Memset"}
CPU_OP = {"cpu_op", "cpu_instant_event", "user_annotation"}


def find_trace(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        cands = sorted(
            [*p.glob("*.pt.trace.json*"), *p.glob("*.json.gz"), *p.glob("*.json")],
            key=lambda f: f.stat().st_mtime,
        )
        if not cands:
            raise SystemExit(f"no trace file in {p}")
        return cands[-1]
    return p


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def aggregate(events: list[dict]) -> dict[str, dict[str, tuple[float, int]]]:
    """bucket -> name -> (total_us, count)."""
    out: dict[str, dict[str, list]] = {
        "gpu_kernel": defaultdict(lambda: [0.0, 0]),
        "memcpy": defaultdict(lambda: [0.0, 0]),
        "cpu_op": defaultdict(lambda: [0.0, 0]),
    }
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "?")
        dur = float(e["dur"])
        if cat in GPU_KERNEL:
            b = "gpu_kernel"
        elif cat in MEMCPY:
            b = "memcpy"
        elif cat in CPU_OP:
            b = "cpu_op"
        else:
            continue
        out[b][name][0] += dur
        out[b][name][1] += 1
    return {b: {n: (v[0], v[1]) for n, v in d.items()} for b, d in out.items()}


def total(bucket: dict[str, tuple[float, int]]) -> float:
    return sum(v[0] for v in bucket.values())


def show(label: str, agg: dict, topn: int = 15) -> None:
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    for b in ("gpu_kernel", "memcpy", "cpu_op"):
        tot = total(agg[b])
        print(f"\n-- {b}  total={tot/1000:.1f} ms --")
        rows = sorted(agg[b].items(), key=lambda kv: -kv[1][0])[:topn]
        for name, (us, cnt) in rows:
            print(f"  {us/1000:9.2f} ms  x{cnt:<6d}  {name[:74]}")


def diff(a_label, a, b_label, b) -> None:
    print(f"\n{'#'*78}\nDIFF  ({b_label}) - ({a_label})   [+ = added by capture]\n{'#'*78}")
    for bucket in ("gpu_kernel", "memcpy", "cpu_op"):
        names = set(a[bucket]) | set(b[bucket])
        deltas = []
        for n in names:
            au = a[bucket].get(n, (0.0, 0))[0]
            bu = b[bucket].get(n, (0.0, 0))[0]
            deltas.append((bu - au, n, au, bu))
        deltas.sort(key=lambda t: -t[0])
        ta, tb = total(a[bucket]), total(b[bucket])
        print(f"\n-- {bucket}  {a_label}={ta/1000:.1f}ms  {b_label}={tb/1000:.1f}ms  "
              f"delta={(tb-ta)/1000:+.1f}ms --")
        for d, n, au, bu in deltas[:15]:
            if abs(d) < 50:  # <0.05 ms noise
                continue
            tag = "NEW" if au == 0 else "   "
            print(f"  {d/1000:+9.2f} ms {tag}  {n[:66]}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    labels = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        tp = find_trace(path)
        print(f"[{label}] {tp}")
        labels.append((label, aggregate(load(tp))))
    for label, agg in labels:
        show(label, agg)
    if len(labels) == 2:
        diff(labels[0][0], labels[0][1], labels[1][0], labels[1][1])


if __name__ == "__main__":
    main()
