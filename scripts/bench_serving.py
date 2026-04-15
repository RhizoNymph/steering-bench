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
) -> dict:
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
    return s


def build_extra_bodies(
    num_prompts: int,
    mode: str,
    shared_vectors,
    diverse_vectors: list,
) -> list[dict | None]:
    if mode in ("disabled", "enabled_idle"):
        return [None] * num_prompts
    if mode == "all_steered_shared":
        return [{"steering_vectors": shared_vectors}] * num_prompts
    # per_request_nK
    k = len(diverse_vectors)
    return [{"steering_vectors": diverse_vectors[i % k]} for i in range(num_prompts)]


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
        "--sharegpt-path",
        default=None,
        help="Path to ShareGPT_V3_unfiltered_cleaned_split.json (optional). "
        "If unset, synthetic prompts are used.",
    )
    parser.add_argument(
        "--modes",
        default="disabled,enabled_idle,all_steered_shared,per_request_n4,per_request_n16",
        help="Comma-separated subset of modes to run",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - {
        "disabled",
        "enabled_idle",
        "all_steered_shared",
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
    }

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
            ],
            log_path=log_dir / "vllm_serving_disabled.log",
        )
        try:
            asyncio.run(wait_for_server(base_url, args.startup_timeout))
            print("\n[phase 1/2] disabled")
            extra = build_extra_bodies(args.num_prompts, "disabled", shared, [])
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
                )
            )
        finally:
            kill_server(proc)
            gc.collect()
            time.sleep(5)

    # Phase 2: enable-steering server (reused across remaining modes)
    steered_modes = [m for m in modes if m != "disabled"]
    if steered_modes:
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
            ],
            log_path=log_dir / "vllm_serving_enabled.log",
        )
        try:
            asyncio.run(wait_for_server(base_url, args.startup_timeout))
            print(f"\n[phase 2/2] enable_steering, max_configs={args.max_steering_configs}")
            for mode in steered_modes:
                extra = build_extra_bodies(
                    args.num_prompts,
                    mode,
                    shared,
                    diverse_n4 if mode == "per_request_n4" else diverse_n16,
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
                    )
                )
        finally:
            kill_server(proc)

    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
