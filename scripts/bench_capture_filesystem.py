#!/usr/bin/env python3
"""ActivationWriter throughput benchmark.

Drives the filesystem writer (ActivationWriter) directly — no LLM, no
capture manager — to measure raw disk throughput and finalize latency.
Answers: "can the writer keep up with the model?"

Sweep dimensions:
  writer_threads    — thread pool size (1, 2, 4, 8)
  hidden_size       — activation width in float16 (768=opt-125m, 4096=7B, 8192=70B)
  num_requests      — concurrent request captures (32, 128)
  steps_per_request — WriteTask count before FinalizeTask (8, 32)

Metrics per sweep point:
  throughput_mb_s   — total MB written / total wall-clock time
  chunks_per_s      — total WriteTask count / total wall-clock time
  finalize_p50_ms   — median latency from submit_finalize to result=ok
  finalize_p99_ms   — 99th percentile finalize latency
"""

from __future__ import annotations

import argparse
import gc
import itertools
import pathlib
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import print_result_summary, write_result
from steering_bench.timing import compute_stats


def _run_one(
    root: pathlib.Path,
    num_threads: int,
    hidden_size: int,
    num_requests: int,
    steps_per_request: int,
    queue_size: int,
) -> dict:
    from vllm.v1.capture.consumers.filesystem.writer import (
        ActivationWriter,
        FinalizeTask,
        WriteTask,
    )

    writer = ActivationWriter(
        root,
        num_threads=num_threads,
        queue_size=queue_size,
        on_collision="overwrite",
    )

    # Timestamps: key → time of submit_finalize call (float, perf_counter)
    submit_finalize_times: dict[tuple[str, int, str], float] = {}
    finalize_done_times: dict[tuple[str, int, str], float] = {}
    done_event = threading.Event()
    total_to_finalize = num_requests
    finalized_count = 0
    count_lock = threading.Lock()

    def on_status(result):
        nonlocal finalized_count
        if result.status in ("ok", "error"):
            key = result.key
            t = time.perf_counter()
            finalize_done_times[key] = t
            with count_lock:
                finalized_count += 1
                if finalized_count >= total_to_finalize:
                    done_event.set()

    writer.add_status_callback(on_status)

    # Build payloads once: 1 row per step, float16.
    row_bytes = hidden_size * 2  # float16
    payload = torch.randn(1, hidden_size, dtype=torch.float16).numpy().tobytes()
    assert len(payload) == row_bytes

    layer, hook_name = 0, "post_mlp"

    # Emit all tasks.
    t_start = time.perf_counter()

    for req_idx in range(num_requests):
        req_id = f"req_{req_idx:06d}"
        key = (req_id, layer, hook_name)
        req_dir = root / req_id
        req_dir.mkdir(parents=True, exist_ok=True)
        bin_path = req_dir / f"{layer}_{hook_name}.bin"
        sidecar_path = req_dir / f"{layer}_{hook_name}.json"

        for step in range(steps_per_request):
            writer.submit(WriteTask(
                path=bin_path,
                payload=payload,
                append=(step > 0),
                key=key,
            ))

        submit_finalize_times[key] = time.perf_counter()
        writer.submit(FinalizeTask(
            bin_path=bin_path,
            sidecar_path=sidecar_path,
            sidecar_payload={"req_id": req_id, "layer": layer, "hook": hook_name},
            key=key,
        ))

    # Wait for all finalizations.
    completed_in_time = done_event.wait(timeout=120.0)
    t_end = time.perf_counter()

    writer.shutdown(timeout=30.0)

    with count_lock:
        actual_completed = finalized_count
    if not completed_in_time:
        raise RuntimeError(
            f"Timed out: only {actual_completed}/{total_to_finalize} "
            "finalizations completed within 120s"
        )

    total_seconds = t_end - t_start
    total_chunks = num_requests * steps_per_request
    total_bytes = total_chunks * row_bytes

    throughput_mb_s = (total_bytes / (1024 * 1024)) / total_seconds
    chunks_per_s = total_chunks / total_seconds

    # Finalize latencies (only keys that completed).
    finalize_latencies_ms = []
    for key, done_t in finalize_done_times.items():
        submit_t = submit_finalize_times.get(key)
        if submit_t is not None:
            finalize_latencies_ms.append((done_t - submit_t) * 1000.0)

    fin_stats = compute_stats(finalize_latencies_ms) if finalize_latencies_ms else None
    completed = len(finalize_done_times)

    return {
        "throughput_mb_s": throughput_mb_s,
        "chunks_per_s": chunks_per_s,
        "total_mb": total_bytes / (1024 * 1024),
        "total_seconds": total_seconds,
        "completed": completed,
        "finalize_p50_ms": fin_stats.p50_ms if fin_stats else None,
        "finalize_p99_ms": fin_stats.p99_ms if fin_stats else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ActivationWriter throughput and latency"
    )
    parser.add_argument(
        "--writer-threads", default="1,2,4,8",
        help="Comma-separated thread pool sizes"
    )
    parser.add_argument(
        "--hidden-sizes", default="768,4096,8192",
        help="Comma-separated hidden sizes (float16 rows)"
    )
    parser.add_argument(
        "--num-requests", default="32,128",
        help="Comma-separated concurrent request counts"
    )
    parser.add_argument(
        "--steps-per-request", default="8,32",
        help="Comma-separated steps (WriteTask count per request before finalize)"
    )
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    writer_threads_list = [int(x) for x in args.writer_threads.split(",")]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    num_requests_list = [int(x) for x in args.num_requests.split(",")]
    steps_list = [int(x) for x in args.steps_per_request.split(",")]

    total = (
        len(writer_threads_list) * len(hidden_sizes)
        * len(num_requests_list) * len(steps_list)
    )
    print("ActivationWriter throughput benchmark")
    print(f"  writer_threads={writer_threads_list}, hidden_sizes={hidden_sizes}")
    print(f"  num_requests={num_requests_list}, steps={steps_list}")
    print(f"  total configs={total}")
    print()

    all_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)

        for threads, hs, num_req, steps in itertools.product(
            writer_threads_list, hidden_sizes, num_requests_list, steps_list
        ):
            label = (
                f"threads={threads} hs={hs} reqs={num_req} steps={steps}"
            )
            print(f"  {label}", flush=True)

            # Each run gets its own subdirectory so results don't collide.
            run_dir = root / f"t{threads}_hs{hs}_r{num_req}_s{steps}"
            run_dir.mkdir()

            try:
                result = _run_one(
                    root=run_dir,
                    num_threads=threads,
                    hidden_size=hs,
                    num_requests=num_req,
                    steps_per_request=steps,
                    queue_size=args.queue_size,
                )
                print(
                    f"    throughput={result['throughput_mb_s']:.1f} MB/s  "
                    f"chunks/s={result['chunks_per_s']:.0f}  "
                    f"fin_p50={result['finalize_p50_ms']:.1f}ms  "
                    f"fin_p99={result['finalize_p99_ms']:.1f}ms  "
                    f"completed={result['completed']}/{num_req}"
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                result = {"error": str(exc)}

            gc.collect()

            all_results.append({
                "writer_threads": threads,
                "hidden_size": hs,
                "num_requests": num_req,
                "steps_per_request": steps,
                **result,
            })

    params = {
        "queue_size": args.queue_size,
        "dtype": "float16",
        "rows_per_chunk": 1,
    }
    write_result(
        benchmark="capture.filesystem",
        parameters=params,
        results={"sweep": all_results},
        output_dir=args.output_dir,
        tag=args.tag,
    )

    # Summary table
    print(f"\n{'=' * 100}")
    print("  ActivationWriter Throughput Benchmark")
    print(f"{'=' * 100}")
    print(
        f"{'threads':>8} {'hs':>6} {'reqs':>5} {'steps':>6} "
        f"{'MB/s':>8} {'chunks/s':>10} {'fin_p50':>10} {'fin_p99':>10}"
    )
    print("-" * 100)
    for r in all_results:
        if "error" in r:
            print(
                f"{r['writer_threads']:>8} {r['hidden_size']:>6} "
                f"{r['num_requests']:>5} {r['steps_per_request']:>6} ERROR"
            )
            continue
        print(
            f"{r['writer_threads']:>8} {r['hidden_size']:>6} "
            f"{r['num_requests']:>5} {r['steps_per_request']:>6} "
            f"{r['throughput_mb_s']:>8.1f} "
            f"{r['chunks_per_s']:>10.0f} "
            f"{r['finalize_p50_ms']:>10.1f} "
            f"{r['finalize_p99_ms']:>10.1f}"
        )
    print(f"{'=' * 100}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
