#!/usr/bin/env python3
"""Summarize a run_kernel_isolation.sh A/B against the steering Triton kernel.

Reads the JSONs produced by bench_cuda_graphs.py under
``--root``/{decode,prefill}/ with tags ``kernel-on`` and ``kernel-off``,
and prints per-cell deltas plus an overall summary.

The interesting numbers:

* ``Δ`` (relative): ``(kernel_off - kernel_on) / kernel_off``. Positive
  means the kernel made the workload faster.
* ``Δ_ms``: ``kernel_off - kernel_on`` absolute.
* Geomean across batch sizes for each label / workload, plus the global
  ``kernel contribution`` headline.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def collect(root: str) -> dict:
    """Return ``{workload: {tag: {(label, bs): median_ms}}}``."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for workload in ("decode", "prefill"):
        wdir = os.path.join(root, workload)
        if not os.path.isdir(wdir):
            continue
        for fp in glob.glob(os.path.join(wdir, "ablation_cuda_graphs_*.json")):
            with open(fp) as f:
                d = json.load(f)
            tag = d.get("tag") or ""
            p = d["parameters"]
            label = p.get("label", "?")
            bs = p.get("batch_size")
            med = d["results"]["latency_ms"]["median_ms"]
            key = (label, bs)
            # if multiple runs for same cell, take the most recent (last)
            out[workload][tag][key] = med
    return out


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def print_table(workload: str, on: dict, off: dict) -> list[float]:
    if not on or not off:
        print(f"\n[{workload}] missing data (on={bool(on)} off={bool(off)})")
        return []
    print(f"\n[{workload}]")
    print(
        f"  {'label':>20s} {'bs':>4s} "
        f"{'kernel-on (ms)':>16s} {'kernel-off (ms)':>16s} "
        f"{'Δ_ms':>10s} {'Δ_rel':>10s}"
    )
    print(f"  {'-' * 80}")
    rels: list[float] = []
    keys = sorted(set(on.keys()) & set(off.keys()))
    # Print all labels: graphs_with_steer is the main one for kernel
    # isolation; the others are scaffolding (no_steer paths and eager
    # paths bracket the measurement).
    for key in keys:
        label, bs = key
        m_on = on[key]
        m_off = off[key]
        d_ms = m_off - m_on
        d_rel = (m_off - m_on) / m_off if m_off > 0 else float("nan")
        if label == "graphs_with_steer":
            rels.append(m_off / m_on)  # ratio for geomean
        print(
            f"  {label:>20s} {bs:>4d} "
            f"{m_on:>16.2f} {m_off:>16.2f} "
            f"{d_ms:>+10.2f} {d_rel * 100:>+9.2f}%"
        )
    if rels:
        print(
            f"\n  graphs_with_steer kernel-off / kernel-on geomean: "
            f"{gmean(rels):.3f}×"
        )
    return rels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="root dir with decode/ prefill/")
    args = ap.parse_args()

    data = collect(args.root)

    all_rels: list[float] = []
    for workload in ("decode", "prefill"):
        on = data.get(workload, {}).get("kernel-on", {})
        off = data.get(workload, {}).get("kernel-off", {})
        rels = print_table(workload, on, off)
        all_rels.extend(rels)

    print(f"\n=== headline ===")
    if all_rels:
        print(
            f"kernel-off / kernel-on geomean across all "
            f"graphs_with_steer cells: {gmean(all_rels):.3f}×"
        )
        print(
            f"i.e. the Triton kernel is ~"
            f"{(gmean(all_rels) - 1) * 100:.1f}% faster than the eager "
            f"body under graph capture, averaged across batch sizes and "
            f"both workloads."
        )
    else:
        print("no graphs_with_steer cells found")


if __name__ == "__main__":
    main()
