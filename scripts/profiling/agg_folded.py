#!/usr/bin/env python3
"""Aggregate a py-spy folded (raw) stack file.

Reports: total samples, inclusive samples under capture-related frames,
and the top self-time (leaf) frames — so we can see how much worker CPU
time the capture dispatch consumes vs the model forward.
"""

from __future__ import annotations

import sys
from collections import defaultdict

CAPTURE_HINTS = (
    "capture", "on_hook", "dispatch_step", "manager.py", "activation_capture",
    "_acquire_pinned", "materialize_global", "index_select", "capture_residual",
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pyspy_lm.folded"
    total = 0
    cap_incl = 0
    self_time: dict[str, int] = defaultdict(int)
    incl_time: dict[str, int] = defaultdict(int)
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, cnt = line.rpartition(" ")
        try:
            c = int(cnt)
        except ValueError:
            continue
        total += c
        frames = stack.split(";")
        leaf = frames[-1] if frames else "?"
        self_time[leaf] += c
        seen = set()
        low = stack.lower()
        if any(h in low for h in CAPTURE_HINTS):
            cap_incl += c
        for f in frames:
            if f not in seen:
                incl_time[f] += c
                seen.add(f)

    print(f"total samples: {total}")
    if total:
        print(f"capture-inclusive samples: {cap_incl} ({cap_incl/total*100:.1f}%)")
    print("\n== top 25 SELF-time (leaf) frames ==")
    for name, c in sorted(self_time.items(), key=lambda kv: -kv[1])[:25]:
        pct = c / total * 100 if total else 0
        print(f"  {pct:5.1f}%  {c:6d}  {name[:88]}")
    print("\n== top 20 capture-related frames by INCLUSIVE samples ==")
    caps = {k: v for k, v in incl_time.items()
            if any(h in k.lower() for h in CAPTURE_HINTS)}
    for name, c in sorted(caps.items(), key=lambda kv: -kv[1])[:20]:
        pct = c / total * 100 if total else 0
        print(f"  {pct:5.1f}%  {c:6d}  {name[:88]}")


if __name__ == "__main__":
    main()
