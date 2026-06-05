#!/usr/bin/env python3
"""ActivationWriter throughput benchmark.

Drives the filesystem writer (ActivationWriter) directly — no LLM, no
capture manager — to measure raw disk throughput and finalize latency.
Answers: "can the writer keep up with the model?"

Sweep dimensions:
  writer_threads    — thread pool size (1, 2, 4, 8, ...)
  hidden_size       — activation width in float16 (768=opt-125m, 4096=7B, 8192=70B)
  num_requests      — concurrent request captures (32, 128)
  steps_per_request — WriteTask count before FinalizeTask (8, 32)
  rows_per_chunk    — rows per WriteTask payload (1, ...); larger payloads
                      amortize per-write syscall/RTT overhead, which is the
                      dominant cost on network filesystems (NFS)

For network mounts, point --bench-dir at the mount and widen the thread
sweep (e.g. 1,2,4,8,16,32): the optimal thread count to hide per-write
round-trip latency sits well past where local disk throughput flattens.
The underlying filesystem type and mount options are auto-detected and
recorded in the result JSON so NFS runs are distinguishable from local disk.

Metrics per sweep point:
  throughput_mb_s   — total MB written / total wall-clock time
  chunks_per_s      — total WriteTask count / total wall-clock time
  finalize_p50_ms   — median latency from submit_finalize to result=ok
  finalize_p99_ms   — 99th percentile finalize latency
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import pathlib
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from steering_bench.output import print_result_summary, write_result
from steering_bench.timing import compute_stats


def _detect_storage(path: pathlib.Path) -> dict:
    """Resolve the filesystem backing ``path`` via /proc/self/mountinfo.

    Returns the fstype (e.g. "nfs4", "ext4", "tmpfs"), the mount source,
    the VFS mount options (where the ro/rw flag that governs whether
    writes succeed lives), and the per-superblock options (which on NFS
    carry rsize/wsize/proto/timeo — the knobs that bound write
    bandwidth). ``read_only`` is derived from the VFS options. Falls back
    to "unknown" rather than raising so the benchmark still runs if the
    proc layout is unexpected.
    """
    target = str(path.resolve())
    best = {
        "fstype": "unknown", "source": "unknown",
        "options": "", "vfs_options": "", "read_only": False,
    }
    best_len = -1
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                # mountinfo: id parent maj:min root mount_point vfs_opts
                #            [optional...] - fstype source super_opts
                parts = line.split()
                try:
                    sep = parts.index("-")
                except ValueError:
                    continue
                mount_point = parts[4]
                vfs_opts = parts[5]
                fstype = parts[sep + 1]
                source = parts[sep + 2]
                super_opts = parts[sep + 3] if len(parts) > sep + 3 else ""
                if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) \
                        and len(mount_point) > best_len:
                    best_len = len(mount_point)
                    best = {
                        "fstype": fstype,
                        "source": source,
                        "options": super_opts,
                        "vfs_options": vfs_opts,
                        "read_only": "ro" in vfs_opts.split(","),
                        "mount_point": mount_point,
                    }
    except OSError:
        pass
    return best


def _run_one(
    root: pathlib.Path,
    num_threads: int,
    hidden_size: int,
    num_requests: int,
    steps_per_request: int,
    queue_size: int,
    rows_per_chunk: int,
    fsync: bool,
    atomic_publish: bool,
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
        fsync=fsync,
        atomic_publish=atomic_publish,
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

    # Build payloads once: rows_per_chunk rows per step, float16.
    row_bytes = hidden_size * 2  # float16
    chunk_bytes = row_bytes * rows_per_chunk
    payload = torch.randn(
        rows_per_chunk, hidden_size, dtype=torch.float16
    ).numpy().tobytes()
    assert len(payload) == chunk_bytes

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
    total_bytes = total_chunks * chunk_bytes

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
        "--hidden-sizes", default="2560,4096,8192",
        help=(
            "Comma-separated hidden sizes (float16 rows). "
            "Defaults cover gemma-3-4b-it (2560), llama-7B-class (4096), "
            "llama-70B-class (8192)."
        ),
    )
    parser.add_argument(
        "--num-requests", default="32,128",
        help="Comma-separated concurrent request counts"
    )
    parser.add_argument(
        "--steps-per-request", default="8,32",
        help="Comma-separated steps (WriteTask count per request before finalize)"
    )
    parser.add_argument(
        "--rows-per-chunk", default="1",
        help=(
            "Comma-separated rows per WriteTask payload. Larger values "
            "amortize per-write syscall/RTT overhead; sweep this on NFS "
            "(e.g. 1,8,32) to find the write size that saturates the mount."
        ),
    )
    parser.add_argument(
        "--fsync", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "fsync each .bin/.json before the atomic rename (durable). "
            "Use --no-fsync to skip all fsyncs — trades crash-durability "
            "for throughput, the dominant finalize cost on network mounts."
        ),
    )
    parser.add_argument(
        "--atomic-publish", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Publish via .tmp + atomic rename (default). Use "
            "--no-atomic-publish to write straight to the final path, "
            "dropping two rename RPCs per capture — the main "
            "small-capture throughput lever on NFS."
        ),
    )
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="results/capture/")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--bench-dir",
        default=None,
        help=(
            "Directory on real disk to write benchmark data into. "
            "Each run gets a cleaned-up subdirectory. "
            "Omit to use a tmpfs temp dir (fast, but does not reflect "
            "real SSD/HDD throughput)."
        ),
    )
    args = parser.parse_args()

    writer_threads_list = [int(x) for x in args.writer_threads.split(",")]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    num_requests_list = [int(x) for x in args.num_requests.split(",")]
    steps_list = [int(x) for x in args.steps_per_request.split(",")]
    rows_per_chunk_list = [int(x) for x in args.rows_per_chunk.split(",")]

    total = (
        len(writer_threads_list) * len(hidden_sizes)
        * len(num_requests_list) * len(steps_list)
        * len(rows_per_chunk_list)
    )
    using_tmpfs = args.bench_dir is None
    print("ActivationWriter throughput benchmark")
    print(f"  writer_threads={writer_threads_list}, hidden_sizes={hidden_sizes}")
    print(f"  num_requests={num_requests_list}, steps={steps_list}")
    print(f"  rows_per_chunk={rows_per_chunk_list}")
    print(f"  total configs={total}")
    storage = {"fstype": "tmpfs", "source": "tmpfs", "options": ""}
    if using_tmpfs:
        print("  WARNING: writing to tmpfs (in-memory). Use --bench-dir for real disk I/O.")
    else:
        bench_path = pathlib.Path(args.bench_dir)
        storage = _detect_storage(bench_path)
        print(
            f"  bench_dir={args.bench_dir} "
            f"(fstype={storage['fstype']} source={storage['source']})"
        )
        if storage["options"]:
            print(f"  mount_options={storage['options']}")
        if storage.get("read_only"):
            print(
                f"  ERROR: {storage.get('mount_point', args.bench_dir)} is "
                "mounted read-only (vfs_options="
                f"{storage['vfs_options']}); cannot run a write benchmark "
                "here. Remount rw or pick a writable --bench-dir.",
                file=sys.stderr,
            )
            sys.exit(2)
        bench_path.mkdir(parents=True, exist_ok=True)
    print()

    all_results = []

    @contextlib.contextmanager
    def _bench_root():
        if using_tmpfs:
            with tempfile.TemporaryDirectory() as d:
                yield pathlib.Path(d)
        else:
            p = pathlib.Path(args.bench_dir)
            p.mkdir(parents=True, exist_ok=True)
            try:
                yield p
            finally:
                shutil.rmtree(p, ignore_errors=True)

    with _bench_root() as root:

        for threads, hs, num_req, steps, rows in itertools.product(
            writer_threads_list, hidden_sizes, num_requests_list, steps_list,
            rows_per_chunk_list,
        ):
            label = (
                f"threads={threads} hs={hs} reqs={num_req} "
                f"steps={steps} rows={rows}"
            )
            print(f"  {label}", flush=True)

            # Each run gets its own subdirectory so results don't collide.
            run_dir = root / f"t{threads}_hs{hs}_r{num_req}_s{steps}_rc{rows}"
            run_dir.mkdir()

            try:
                result = _run_one(
                    root=run_dir,
                    num_threads=threads,
                    hidden_size=hs,
                    num_requests=num_req,
                    steps_per_request=steps,
                    queue_size=args.queue_size,
                    rows_per_chunk=rows,
                    fsync=args.fsync,
                    atomic_publish=args.atomic_publish,
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
                "rows_per_chunk": rows,
                **result,
            })

    params = {
        "queue_size": args.queue_size,
        "dtype": "float16",
        "rows_per_chunk_sweep": rows_per_chunk_list,
        "fsync": args.fsync,
        "atomic_publish": args.atomic_publish,
        "bench_storage": storage["fstype"],
        "bench_source": storage["source"],
        "bench_mount_options": storage["options"],
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
        f"{'threads':>8} {'hs':>6} {'reqs':>5} {'steps':>6} {'rows':>5} "
        f"{'MB/s':>8} {'chunks/s':>10} {'fin_p50':>10} {'fin_p99':>10}"
    )
    print("-" * 100)
    for r in all_results:
        if "error" in r:
            print(
                f"{r['writer_threads']:>8} {r['hidden_size']:>6} "
                f"{r['num_requests']:>5} {r['steps_per_request']:>6} "
                f"{r['rows_per_chunk']:>5} ERROR"
            )
            continue
        print(
            f"{r['writer_threads']:>8} {r['hidden_size']:>6} "
            f"{r['num_requests']:>5} {r['steps_per_request']:>6} "
            f"{r['rows_per_chunk']:>5} "
            f"{r['throughput_mb_s']:>8.1f} "
            f"{r['chunks_per_s']:>10.0f} "
            f"{r['finalize_p50_ms']:>10.1f} "
            f"{r['finalize_p99_ms']:>10.1f}"
        )
    print(f"{'=' * 100}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
