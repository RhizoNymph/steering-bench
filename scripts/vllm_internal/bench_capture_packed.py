#!/usr/bin/env python3
"""per_file vs packed layout benchmark (consumer-driven).

Unlike bench_capture_filesystem.py (which drives ActivationWriter directly),
this drives the full ``FilesystemConsumer`` so the ``packed`` layout — which
lives in the consumer — is exercised. The packed win only appears with
multiple layers per request: ``per_file`` writes one file per (layer, hook),
``packed`` writes one file per request, cutting metadata RPCs by num_layers.

Each request captures ``num_layers`` layers of one hook, ``steps`` rows each
(one row per decode step). Reports throughput (MB/s), on-disk file count, and
finalize latency, per layout.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from steering_bench.output import write_result


def _run_one(
    root: pathlib.Path,
    layout: str,
    num_requests: int,
    num_layers: int,
    steps: int,
    hidden: int,
    writer_threads: int,
    fsync: bool,
    atomic_publish: bool,
    coalesce_max_bytes: int,
    batch_submit: bool = False,
) -> dict:
    from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
    from vllm.v1.capture.consumers.filesystem.types import FilesystemCaptureRequest
    from vllm.v1.capture.types import (
        CaptureChunk,
        CaptureContext,
        CaptureFinalize,
        VllmInternalRequestId,
    )

    hook = "post_block"
    layers = list(range(num_layers))
    consumer = FilesystemConsumer(
        vllm_config=MagicMock(),
        params={
            "root": str(root),
            "writer_threads": writer_threads,
            "queue_size": 8192,
            "fsync": fsync,
            "atomic_publish": atomic_publish,
            "default_layout": layout,
            "coalesce_max_bytes": coalesce_max_bytes,
        },
    )

    # Pre-build one row tensor per (step) reused across layers/requests;
    # bytes are what matter for I/O.
    row = torch.randn(1, hidden, dtype=torch.float32)
    row_bytes = hidden * 4
    req_ids = [f"req_{i:05d}" for i in range(num_requests)]

    # Admission: register every request (records layout + expected keys).
    for rid in req_ids:
        raw = FilesystemCaptureRequest(
            request_id=rid, tag="bench",
            hooks={hook: layers}, positions="last_prompt", layout=layout,
        )
        ctx = CaptureContext(
            vllm_internal_request_id=VllmInternalRequestId(rid),
            num_prompt_tokens=8, num_computed_tokens=0,
            num_hidden_layers=num_layers, hidden_size=hidden,
            element_size_bytes=4, tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        consumer.validate_client_spec(raw, ctx)

    all_keys = [
        (VllmInternalRequestId(rid), layer, hook)
        for rid in req_ids for layer in layers
    ]

    t_start = time.perf_counter()
    # Submit chunks step-by-step, interleaved across requests and layers
    # (mirrors per-step dispatch).
    for step in range(steps):
        for rid in req_ids:
            if batch_submit:
                # Mirror the manager: hand a request's whole step (all
                # layers) to the consumer in one batch call.
                consumer.submit_chunk_batch(
                    [
                        CaptureChunk(
                            key=(VllmInternalRequestId(rid), layer, hook),
                            tensor=row, dtype=row.dtype, row_offset=step,
                            step_index=step, metadata={},
                        )
                        for layer in layers
                    ]
                )
            else:
                for layer in layers:
                    consumer.submit_chunk(
                        CaptureChunk(
                            key=(VllmInternalRequestId(rid), layer, hook),
                            tensor=row, dtype=row.dtype, row_offset=step,
                            step_index=step, metadata={},
                        )
                    )
    # Per-request finalize burst (all of a request's keys).
    submit_finalize_t: dict = {}
    for rid in req_ids:
        for layer in layers:
            submit_finalize_t[(rid, layer)] = time.perf_counter()
            consumer.submit_finalize(
                CaptureFinalize(
                    key=(VllmInternalRequestId(rid), layer, hook), sidecar={}
                )
            )

    # Wait for every key to reach a terminal status.
    fin_latencies_ms: list[float] = []
    completed = 0
    for key in all_keys:
        r = consumer.wait_for_result(key, timeout=120.0)
        done_t = time.perf_counter()
        rid = str(key[0])
        st = submit_finalize_t.get((rid, key[1]))
        if st is not None:
            fin_latencies_ms.append((done_t - st) * 1000.0)
        if r is not None and r.status == "ok":
            completed += 1
    t_end = time.perf_counter()
    consumer.shutdown(timeout=30.0)

    total_bytes = num_requests * num_layers * steps * row_bytes
    total_seconds = t_end - t_start
    throughput_mb_s = (total_bytes / 1e6) / total_seconds

    # Count files actually on disk. per_file/sharded use .bin; the packed
    # layout writes one self-contained .packed file per request (index
    # inlined as a trailer, no separate sidecar).
    data_files = list(root.rglob("*.bin")) + list(root.rglob("*.packed"))
    file_count = len(data_files)

    fin_latencies_ms.sort()
    p50 = (
        fin_latencies_ms[len(fin_latencies_ms) // 2] if fin_latencies_ms else 0.0
    )
    p99 = (
        fin_latencies_ms[min(len(fin_latencies_ms) - 1,
                             int(len(fin_latencies_ms) * 0.99))]
        if fin_latencies_ms else 0.0
    )

    return {
        "layout": layout,
        "throughput_mb_s": throughput_mb_s,
        "total_mb": total_bytes / 1e6,
        "total_seconds": total_seconds,
        "file_count": file_count,
        "completed_keys": completed,
        "total_keys": len(all_keys),
        "finalize_p50_ms": p50,
        "finalize_p99_ms": p99,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="per_file vs packed layout benchmark")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--layouts", default="per_file,packed")
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--writer-threads", type=int, default=8)
    p.add_argument("--fsync", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--atomic-publish", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--coalesce-max-bytes", type=int, default=1 << 20)
    p.add_argument("--batch-submit", action="store_true",
                   help="submit a request's per-step layers via submit_chunk_batch")
    p.add_argument("--output-dir", default="results/capture/")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    bench_dir = pathlib.Path(args.bench_dir)
    results = []
    print(
        f"packed-layout benchmark: reqs={args.num_requests} "
        f"layers={args.num_layers} steps={args.steps} hidden={args.hidden} "
        f"threads={args.writer_threads} fsync={args.fsync} "
        f"atomic={args.atomic_publish}"
    )
    print(f"{'layout':>9} {'MB/s':>8} {'files':>7} {'fin_p50':>9} {'fin_p99':>9} "
          f"{'ok/total':>10}")
    print("-" * 64)
    for layout in args.layouts.split(","):
        run_dir = bench_dir / layout
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = _run_one(
                root=run_dir, layout=layout, num_requests=args.num_requests,
                num_layers=args.num_layers, steps=args.steps, hidden=args.hidden,
                writer_threads=args.writer_threads, fsync=args.fsync,
                atomic_publish=args.atomic_publish,
                coalesce_max_bytes=args.coalesce_max_bytes,
                batch_submit=args.batch_submit,
            )
            print(f"{layout:>9} {r['throughput_mb_s']:>8.1f} {r['file_count']:>7} "
                  f"{r['finalize_p50_ms']:>9.1f} {r['finalize_p99_ms']:>9.1f} "
                  f"{r['completed_keys']}/{r['total_keys']:>}")
            results.append(r)
        except Exception as exc:
            print(f"{layout:>9} ERROR: {exc}")
            results.append({"layout": layout, "error": str(exc)})
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    write_result(
        benchmark="capture.packed",
        parameters={
            "num_requests": args.num_requests, "num_layers": args.num_layers,
            "steps": args.steps, "hidden": args.hidden,
            "writer_threads": args.writer_threads, "fsync": args.fsync,
            "atomic_publish": args.atomic_publish, "dtype": "float32",
        },
        results={"sweep": results},
        output_dir=args.output_dir, tag=args.tag,
    )


if __name__ == "__main__":
    main()
