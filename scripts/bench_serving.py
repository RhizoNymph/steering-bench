#!/usr/bin/env python3
"""Online serving benchmark: TTFT, TPOT, ITL, E2EL across steering modes.

Launches ``vllm serve`` as a subprocess and drives it through the
OpenAI-compatible HTTP API. Measures time-to-first-token (TTFT),
time-per-output-token (TPOT), inter-token latency (ITL), and end-to-end
latency (E2EL) for each mode. This fills the "online serving" gap in the
existing offline-only benchmark suite.

Modes:
    disabled             server started without --enable-steering
    enabled_idle         steering enabled, no vectors in requests
    all_steered_shared   every request uses the same steering vector
    per_request_n4       4 distinct configs spread across requests
    per_request_n16      16 distinct configs spread across requests

Workloads:
    synthetic (default)  fixed-length prompts generated locally
    sharegpt             via --sharegpt-path pointing at a local ShareGPT_V3 json

Each mode runs a discarded warmup pass (defaults to --concurrency requests
at --warmup-max-tokens=8) before measurement so Triton JIT compile, the
first-touch auto-promote LRU, and the H2D staging path do not dominate the
first wave of measured requests. After warmup, the bench sleeps
--warmup-drain-seconds (default 0.5 s) and fires one 1-token soft-barrier
request so async background work (broadcasts, deferred H2D) queued during
warmup lands before measurement begins — without that drain, PR worktrees
with async-dispatch behavior showed a phantom +2 ms per-token TPOT
regression on the measured pass. Disable warmup with --warmup-requests 0;
disable just the drain with --warmup-drain-seconds 0.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import random_steering_vectors, random_steering_vectors_diverse

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
}


@dataclass
class RequestResult:
    ttft_ms: float | None = None
    e2el_ms: float | None = None
    num_output_tokens: int = 0
    itl_ms: list[float] = field(default_factory=list)
    error: str | None = None


def load_sharegpt(path: Path, num_prompts: int, min_words: int, max_words: int) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    prompts: list[str] = []
    for entry in data:
        conv = entry.get("conversations") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("from") != "human":
            continue
        text = first.get("value", "").strip()
        n_words = len(text.split())
        if min_words <= n_words <= max_words:
            prompts.append(text)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"ShareGPT only yielded {len(prompts)} prompts matching "
            f"{min_words}-{max_words} words; needed {num_prompts}"
        )
    return prompts


def make_synthetic_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def launch_server(
    python_bin: str,
    model: str,
    port: int,
    extra_args: list[str],
    log_path: Path,
    env: dict | None = None,
) -> subprocess.Popen:
    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ] + extra_args
    print(f"[server] launch: {' '.join(cmd)}")
    print(f"[server] log:    {log_path}")
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env or os.environ.copy(),
    )
    return proc


def kill_server(proc: subprocess.Popen, grace: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    print(f"[server] terminate pid={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    print(f"[server] kill -9 pid={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print(f"[server] WARNING: pid={proc.pid} did not die")


async def wait_for_server(base_url: str, timeout: float) -> None:
    import httpx

    deadline = time.perf_counter() + timeout
    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get(f"{base_url}/models")
                if r.status_code == 200:
                    print(f"[server] ready at {base_url}")
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(2.0)
    raise RuntimeError(
        f"server {base_url} not ready within {timeout}s (last error: {last_err})"
    )


async def run_one_request(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_body: dict | None,
) -> RequestResult:
    result = RequestResult()
    last_tok_t: float | None = None
    try:
        kwargs: dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.perf_counter()
        stream = await client.completions.create(**kwargs)
        async for chunk in stream:
            now = time.perf_counter()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].text
            if not delta:
                continue
            if result.ttft_ms is None:
                result.ttft_ms = (now - t0) * 1000.0
                last_tok_t = now
            else:
                if last_tok_t is not None:
                    result.itl_ms.append((now - last_tok_t) * 1000.0)
                last_tok_t = now
            result.num_output_tokens += 1
        result.e2el_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"
    return result


async def run_workload(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    extra_bodies: list[dict | None],
    concurrency: int,
) -> list[RequestResult]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="unused")
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(p: str, eb: dict | None) -> RequestResult:
        async with sem:
            return await run_one_request(client, model, p, max_tokens, eb)

    tasks = [_guarded(p, eb) for p, eb in zip(prompts, extra_bodies, strict=True)]
    return await asyncio.gather(*tasks)


def summarize(results: list[RequestResult]) -> dict:
    ok = [r for r in results if r.error is None and r.ttft_ms is not None]
    errs = [r.error for r in results if r.error is not None]
    if not ok:
        return {
            "num_ok": 0,
            "num_err": len(errs),
            "errors": errs[:5],
        }

    ttft = [r.ttft_ms for r in ok]
    e2el = [r.e2el_ms for r in ok]
    all_itl = [v for r in ok for v in r.itl_ms]
    tpot = [
        (r.e2el_ms - r.ttft_ms) / max(1, r.num_output_tokens - 1)
        for r in ok
        if r.num_output_tokens > 1
    ]
    total_out = sum(r.num_output_tokens for r in ok)
    wall_s = max(e2el) / 1000.0 if e2el else 0.0

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {}
        d = compute_stats(xs).to_dict()
        return {k: v for k, v in d.items() if k != "samples_ms"}

    return {
        "num_ok": len(ok),
        "num_err": len(errs),
        "errors": errs[:5],
        "ttft_ms": _stats(ttft),
        "tpot_ms": _stats(tpot),
        "itl_ms": _stats(all_itl),
        "e2el_ms": _stats(e2el),
        "total_output_tokens": total_out,
        "offline_output_tps": total_out / wall_s if wall_s > 0 else 0.0,
    }


def print_summary(mode: str, n: int, s: dict) -> None:
    if s.get("num_ok", 0) == 0:
        print(f"  {mode}: ALL FAILED ({s.get('num_err', 0)} errors)")
        return
    def m(section: str) -> str:
        d = s.get(section, {})
        return f"{d.get('median_ms', float('nan')):.1f}"
    print(
        f"  {mode:<22} n={n} "
        f"TTFT={m('ttft_ms')}ms "
        f"TPOT={m('tpot_ms')}ms "
        f"ITL={m('itl_ms')}ms "
        f"E2EL={m('e2el_ms')}ms "
        f"throughput={s.get('offline_output_tps', 0):.0f}tok/s"
    )


async def run_mode(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    extra_bodies: list[dict | None],
    concurrency: int,
    mode: str,
    parameters: dict,
    output_dir: str,
    tag: str,
    warmup_requests: int = 0,
    warmup_max_tokens: int = 8,
    warmup_drain_seconds: float = 0.5,
) -> dict:
    # Triton kernels JIT-compile on first invocation with a given
    # specialization, and the auto-promote / inline paths allocate on
    # first-touch. Without a warmup pass the first wave of concurrent
    # requests in the first steered mode pays those costs and inflates
    # TTFT medians by hundreds of ms.
    if warmup_requests > 0:
        n = min(warmup_requests, len(prompts))
        print(f"  {mode}: warmup ({n} reqs, max_tokens={warmup_max_tokens})")
        warm_t0 = time.perf_counter()
        warm = await run_workload(
            base_url,
            model,
            prompts[:n],
            warmup_max_tokens,
            extra_bodies[:n],
            concurrency,
        )
        warm_errs = sum(1 for r in warm if r.error is not None)
        print(
            f"    warmup done in {time.perf_counter() - warm_t0:.1f}s "
            f"({n - warm_errs} ok / {warm_errs} err)"
        )
        # Drain async background work (auto-promote broadcasts,
        # non-blocking H2D, deferred bookkeeping) before measurement.
        # Without this, fire-and-forget tasks queued during warmup
        # land during the measured pass and inflate TPOT — observed
        # as a phantom +2 ms per-token regression on PR worktrees vs
        # base that disappeared once warmup_requests=0.
        # The sleep gives the bookkeeping wall time to land; the
        # single-request soft barrier then ensures any work the
        # sleep didn't catch either completes within it or queues
        # behind it before measurement begins.
        if warmup_drain_seconds > 0:
            print(
                f"    draining async work ({warmup_drain_seconds}s sleep "
                f"+ 1-token barrier)"
            )
            await asyncio.sleep(warmup_drain_seconds)
            await run_workload(
                base_url, model, prompts[:1], 1, extra_bodies[:1], 1
            )
        # Discard timing accumulators so the measured run's per-mode
        # steering-timing dump reflects only the measured requests.
        await dump_and_reset_steering_timings(base_url, mode, quiet=True)

    results = await run_workload(
        base_url, model, prompts, max_tokens, extra_bodies, concurrency
    )
    s = summarize(results)
    print_summary(mode, len(prompts), s)
    params = {**parameters, "mode": mode}
    write_result(
        benchmark="vllm.serving",
        parameters=params,
        results=s,
        output_dir=output_dir,
        tag=tag,
    )
    # No-op unless the server was launched with both
    # VLLM_STEERING_TIMING=1 and VLLM_SERVER_DEV_MODE=1.
    await dump_and_reset_steering_timings(base_url, mode)
    return s


NAMED_BENCH_MODULE = "bench_named_shared"


def pack_steering_vectors(vecs: dict, with_scales: bool = False) -> dict:
    """Convert a dict[hook, dict[layer, list[float]]] to the binary wire
    form used by vllm.config.steering_types.SteeringHookPacked.

    Used only when --packed-vectors is set on the bench. Server-side
    decode requires a vllm build that includes the binary-wire support
    (perf/steering-binary-wire or later).

    When *with_scales* is True, attach a deterministic per-layer ``scales``
    list (varies row to row, all != 1.0) so the server exercises the
    per-row multiply path in ``unpack_steering_vectors``. Requires PR
    #163 (per-layer scales in binary wire) server-side.
    """
    import base64
    import numpy as np

    out: dict[str, dict] = {}
    for hook, layer_dict in vecs.items():
        layer_indices = sorted(layer_dict.keys())
        arr = np.stack(
            [np.asarray(layer_dict[i], dtype=np.float32) for i in layer_indices]
        )
        entry: dict = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "layer_indices": layer_indices,
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        if with_scales:
            # Deterministic per-row scales that vary across rows and are
            # all != 1.0 so every row hits the per-row multiply path.
            entry["scales"] = [
                round(0.5 + 0.05 * i, 4) for i in range(len(layer_indices))
            ]
        out[hook] = entry
    return out


def distinct_configs_for_mode(mode: str, diverse_vectors: list) -> int:
    """Return how many distinct steering configs the mode exercises.

    Used as a floor on the warmup request count so that every config the
    measured pass will see has been first-touched (Triton JIT, auto-promote
    LRU, pinned-buffer alloc) before measurement begins. Without this,
    per_request_n16 with 8 warmup reqs leaves 8 configs cold and pollutes
    the measured tail.
    """
    if mode in ("disabled", "enabled_idle"):
        return 0
    if mode in ("named_shared", "all_steered_shared"):
        return 1
    # per_request_nK rotates through the full diverse list.
    return len(diverse_vectors)


def build_extra_bodies(
    num_prompts: int,
    mode: str,
    shared_vectors,
    diverse_vectors: list,
    packed: bool = False,
    packed_with_scales: bool = False,
) -> list[dict | None]:
    if mode in ("disabled", "enabled_idle"):
        return [None] * num_prompts
    field = "steering_vectors_packed" if packed else "steering_vectors"
    if packed:
        pack = lambda v: pack_steering_vectors(v, with_scales=packed_with_scales)
    else:
        pack = lambda v: v
    if mode == "all_steered_shared":
        packed_shared = pack(shared_vectors)
        return [{field: packed_shared}] * num_prompts
    if mode == "named_shared":
        # Server-side named module pre-registered via
        # /v1/steering/modules/register; only the name (16 bytes-ish) rides
        # the wire per request.
        return [{"steering_name": NAMED_BENCH_MODULE}] * num_prompts
    # per_request_nK — pre-pack each distinct config once so the
    # repeated cycling doesn't re-encode bytes.
    k = len(diverse_vectors)
    packed_diverse = [pack(v) for v in diverse_vectors]
    return [{field: packed_diverse[i % k]} for i in range(num_prompts)]


async def dump_and_reset_steering_timings(
    base_url: str,
    mode: str,
    timeout: float = 30.0,
    quiet: bool = False,
) -> None:
    """Pull host-side steering timing breakdown from each worker.

    Only fires when ``VLLM_STEERING_TIMING=1`` is set in the *server*
    environment.  Hits ``/v1/steering/_timings/dump_and_reset`` (which
    requires ``VLLM_SERVER_DEV_MODE=1``), prints one table per worker
    annotated with the just-finished mode, then resets the accumulators
    so the next mode gets a clean slate.  Pass ``quiet=True`` to reset
    silently — used between warmup and measurement so warmup costs do
    not appear in the per-mode timing table.
    """
    import httpx

    url = base_url.rstrip("/").removesuffix("/v1")
    endpoint = f"{url}/v1/steering/_timings/dump_and_reset"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(endpoint)
    except Exception as e:
        if not quiet:
            print(f"[timing] dump endpoint unavailable ({e}) — skipping")
        return
    if r.status_code == 404:
        # Server not in dev mode — endpoint not attached.
        return
    if r.status_code != 200:
        if not quiet:
            print(f"[timing] dump endpoint returned {r.status_code}: {r.text[:200]}")
        return
    if quiet:
        return
    workers = r.json().get("workers", [])
    if not workers:
        return
    print(f"\n[timing mode={mode}] per-worker steering breakdown")
    for i, worker in enumerate(workers):
        if not worker:
            continue
        name_w = max(len(row[0]) for row in worker)
        print(
            f"  worker[{i}]  "
            f"{'name':<{name_w}}  {'n':>8}  {'total_ms':>12}  "
            f"{'mean_us':>10}  {'max_ms':>10}"
        )
        for entry in worker:
            name, count, total_ns, max_ns = entry
            total_ms = total_ns / 1e6
            mean_us = total_ns / count / 1e3 if count else 0.0
            max_ms = max_ns / 1e6
            print(
                f"  worker[{i}]  "
                f"{name:<{name_w}}  {count:>8d}  {total_ms:>12.3f}  "
                f"{mean_us:>10.2f}  {max_ms:>10.3f}"
            )


async def register_named_module(
    base_url: str,
    name: str,
    vectors: dict,
    timeout: float = 60.0,
) -> None:
    """POST a named steering module to the server's dev-mode registry.

    Requires the server to have been launched with
    ``VLLM_SERVER_DEV_MODE=1`` so
    ``vllm.entrypoints.serve.steering.modules_router`` is attached.
    """
    import httpx

    url = base_url.rstrip("/").removesuffix("/v1")
    endpoint = f"{url}/v1/steering/modules/register"
    payload = {
        "name": name,
        "vectors": vectors,
        "prefill_vectors": None,
        "decode_vectors": None,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(endpoint, json=payload)
        if r.status_code != 200:
            raise RuntimeError(
                f"register_named_module(name={name}) failed: "
                f"{r.status_code} {r.text}"
            )
        print(f"[server] registered named module: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online serving benchmark")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/serving/")
    parser.add_argument("--tag", default="")
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-steering-configs", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help="Discarded warmup requests fired per mode before measurement. "
        "Defaults to --concurrency so every in-flight slot is primed. "
        "Set 0 to disable.",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=8,
        help="max_tokens for each warmup request (default: 8).",
    )
    parser.add_argument(
        "--packed-vectors",
        action="store_true",
        help="Send inline vectors via steering_vectors_packed (binary "
        "wire format) instead of the legacy list-of-floats JSON form. "
        "Requires a vllm build that supports the packed schema "
        "(perf/steering-binary-wire or later). No-op for named_shared, "
        "enabled_idle, and disabled modes.",
    )
    parser.add_argument(
        "--packed-with-scales",
        action="store_true",
        help="When --packed-vectors is set, attach a deterministic "
        "per-layer scales list to each packed hook so the server "
        "exercises the per-row multiply path. Requires PR #163.",
    )
    parser.add_argument(
        "--warmup-drain-seconds",
        type=float,
        default=0.5,
        help="Seconds to sleep after warmup before the soft-barrier "
        "request. Lets async work (broadcasts, deferred H2D) queued "
        "during warmup land before measurement. Set 0 to disable.",
    )
    parser.add_argument(
        "--sharegpt-path",
        default=None,
        help="Path to ShareGPT_V3_unfiltered_cleaned_split.json (optional). "
        "If unset, synthetic prompts are used.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass --enforce-eager to the vLLM server (disables CUDA graph "
        "capture). Used to ablate graph speedup vs steering.",
    )
    parser.add_argument(
        "--modes",
        default="disabled,enabled_idle,named_shared,all_steered_shared,per_request_n4,per_request_n16",
        help="Comma-separated subset of modes to run.  named_shared "
             "pre-registers a single module via "
             "POST /v1/steering/modules/register and references it from "
             "every request — the floor for the spec-reuse case.",
    )
    args = parser.parse_args()

    warmup_requests = (
        args.warmup_requests
        if args.warmup_requests is not None
        else args.concurrency
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - {
        "disabled",
        "enabled_idle",
        "all_steered_shared",
        "named_shared",
        "per_request_n4",
        "per_request_n16",
    }
    if unknown:
        print(f"ERROR: unknown modes: {sorted(unknown)}", file=sys.stderr)
        sys.exit(2)

    model_cfg = MODEL_CONFIGS.get(args.model)
    if model_cfg is None:
        print(
            f"WARNING: unknown model {args.model}, defaulting hidden_size/num_layers "
            f"to Gemma-3-4B values. Add it to MODEL_CONFIGS for correctness.",
            file=sys.stderr,
        )
        model_cfg = {"hidden_size": 2560, "num_layers": 34}

    # Prompts
    if args.sharegpt_path:
        prompts = load_sharegpt(
            Path(args.sharegpt_path),
            num_prompts=args.num_prompts,
            min_words=32,
            max_words=512,
        )
        workload = "sharegpt"
        print(f"loaded {len(prompts)} prompts from ShareGPT")
    else:
        prompts = make_synthetic_prompts(args.num_prompts, args.prompt_len)
        workload = "synthetic"
        print(f"generated {len(prompts)} synthetic prompts (len={args.prompt_len})")

    # Steering vectors
    shared = random_steering_vectors(
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        hook_points=["post_mlp"],
        scale=0.1,
        seed=42,
    )
    diverse_n4 = random_steering_vectors_diverse(
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_configs=4,
        hook_points=["post_mlp"],
        scale=0.1,
        base_seed=100,
    )
    diverse_n16 = random_steering_vectors_diverse(
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        num_configs=16,
        hook_points=["post_mlp"],
        scale=0.1,
        base_seed=200,
    )

    base_url = f"http://127.0.0.1:{args.port}/v1"
    log_dir = Path("/tmp")
    log_dir.mkdir(exist_ok=True)

    parameters_base = {
        "model": args.model,
        "workload": workload,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompt_len": args.prompt_len if workload == "synthetic" else None,
        "max_model_len": args.max_model_len,
        "sharegpt_path": args.sharegpt_path,
        "warmup_requests": warmup_requests,
        "warmup_max_tokens": args.warmup_max_tokens,
        "warmup_drain_seconds": args.warmup_drain_seconds,
        "packed_vectors": args.packed_vectors,
        "packed_with_scales": args.packed_with_scales,
        "enforce_eager": args.enforce_eager,
    }

    eager_args = ["--enforce-eager"] if args.enforce_eager else []

    # Phase 1: disabled (needs its own server)
    if "disabled" in modes:
        proc = launch_server(
            python_bin=args.python_bin,
            model=args.model,
            port=args.port,
            extra_args=[
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ] + eager_args,
            log_path=log_dir / "vllm_serving_disabled.log",
        )
        try:
            asyncio.run(wait_for_server(base_url, args.startup_timeout))
            print("\n[phase 1/2] disabled")
            extra = build_extra_bodies(args.num_prompts, "disabled", shared, [])
            mode_warmup = max(
                warmup_requests, distinct_configs_for_mode("disabled", [])
            )
            asyncio.run(
                run_mode(
                    base_url,
                    args.model,
                    prompts,
                    args.max_tokens,
                    extra,
                    args.concurrency,
                    "disabled",
                    {**parameters_base, "enable_steering": False},
                    args.output_dir,
                    args.tag,
                    warmup_requests=mode_warmup,
                    warmup_max_tokens=args.warmup_max_tokens,
                    warmup_drain_seconds=args.warmup_drain_seconds,
                )
            )
        finally:
            kill_server(proc)
            gc.collect()
            time.sleep(5)

    # Phase 2: enable-steering server (reused across remaining modes)
    steered_modes = [m for m in modes if m != "disabled"]
    if steered_modes:
        # named_shared needs the dev-mode admin endpoint to register
        # the module before requests reference it.
        steered_env = os.environ.copy()
        if "named_shared" in steered_modes:
            steered_env["VLLM_SERVER_DEV_MODE"] = "1"
        proc = launch_server(
            python_bin=args.python_bin,
            model=args.model,
            port=args.port,
            extra_args=[
                "--enable-steering",
                "--max-steering-configs",
                str(args.max_steering_configs),
                "--max-model-len",
                str(args.max_model_len),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ] + eager_args,
            log_path=log_dir / "vllm_serving_enabled.log",
            env=steered_env,
        )
        try:
            asyncio.run(wait_for_server(base_url, args.startup_timeout))
            print(f"\n[phase 2/2] enable_steering, max_configs={args.max_steering_configs}")
            if "named_shared" in steered_modes:
                asyncio.run(
                    register_named_module(base_url, NAMED_BENCH_MODULE, shared)
                )
            for mode in steered_modes:
                diverse_for_mode = (
                    diverse_n4 if mode == "per_request_n4" else diverse_n16
                )
                extra = build_extra_bodies(
                    args.num_prompts,
                    mode,
                    shared,
                    diverse_for_mode,
                    packed=args.packed_vectors,
                    packed_with_scales=args.packed_with_scales,
                )
                mode_warmup = max(
                    warmup_requests,
                    distinct_configs_for_mode(mode, diverse_for_mode),
                )
                asyncio.run(
                    run_mode(
                        base_url,
                        args.model,
                        prompts,
                        args.max_tokens,
                        extra,
                        args.concurrency,
                        mode,
                        {
                            **parameters_base,
                            "enable_steering": True,
                            "max_steering_configs": args.max_steering_configs,
                        },
                        args.output_dir,
                        args.tag,
                        warmup_requests=mode_warmup,
                        warmup_max_tokens=args.warmup_max_tokens,
                        warmup_drain_seconds=args.warmup_drain_seconds,
                    )
                )
        finally:
            kill_server(proc)

    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
