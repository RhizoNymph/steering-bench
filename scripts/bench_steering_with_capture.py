#!/usr/bin/env python3
"""Steering latency with vs. without the capture-consumer plugin system.

The capture-consumer subsystem ships an always-loaded ``maybe_capture_residual``
gate inside ``apply_layer_steering``: when no manager is installed
(``--capture-consumers`` unset on the server, ``capture_consumers=None`` in
the LLM constructor) the call is a Python ``None`` check that ``torch.compile``
is supposed to constant-fold out of the compiled graph. When a manager *is*
installed the gate dispatches to ``torch.ops.vllm.capture_residual`` on every
``(layer, hook)`` invocation regardless of whether any per-request spec asked
for that position — the manager decides whether to materialize the row.

This benchmark measures whether enabling the plugin system actually adds
overhead to the steering hot path. For every steering mode (disabled,
enabled_idle, per_request_1, per_request_4) it sweeps three capture states:

  cap_off       no consumers, capture system inactive (cold path)
  cap_on_idle   filesystem consumer registered but the request does NOT pass
                a per-request ``capture`` field — manager installed, plan
                empty, captures the per-step bookkeeping overhead with no
                actual gather/dispatch work
  cap_on_active same filesystem consumer + per-request capture asking for
                last_prompt at (post_block, layer L) — manager builds a plan,
                gathers rows, dispatches chunks to the writer

The filesystem consumer is used (rather than logging) because it has
``reads_client_spec=True``, which is necessary for the per-request opt-in path
to actually accept the spec at admission. Each ``LLM`` is given a fresh
tempdir so writes don't compound across runs.

Output: same schema as ``bench_latency.py`` but tagged ``vllm.steering_with_capture``
plus the three new per-config columns. Each (mode, cap, batch_size) cell is
run in a fresh subprocess so vLLM's residual weight memory does not leak across
configs.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

# Force vLLM to run the engine core in-process. The default forks a
# subprocess after the parent has touched CUDA (we call
# ``torch.cuda.is_available()`` in ``main`` plus ``torch.cuda.synchronize``
# in the measurement loop), and PyTorch refuses to re-init CUDA in a
# forked child. Mirrors the convention in nsys_target.py /
# profile_steering.py / bench_capture_plugin_work.py.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from steering_bench.capture_consumers.runner import run_in_subprocess
from steering_bench.engine.base import SteeringConfig
from steering_bench.engine.capture import CaptureConsumerSpec
from steering_bench.engine.engines.vllm import VllmSteeringEngine
from steering_bench.engine.spec import GenerationRequest, RequestCapture, SteeringSpec
from steering_bench.harness.args import engine_names
from steering_bench.harness.models import get_model_config
from steering_bench.output import write_result
from steering_bench.timing import compute_stats
from steering_bench.vectors import (
    random_steering_vectors,
    random_steering_vectors_diverse,
)

STEERING_MODES = ("disabled", "enabled_idle", "per_request_1", "per_request_4")
CAPTURE_MODES = ("cap_off", "cap_on_idle", "cap_on_active")


def _make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    return [" ".join(["hello"] * words_needed)] * num_prompts


def _build_capture_consumers(
    capture_mode: str,
    capture_layer: int,
    fs_root: str | None,
) -> list[CaptureConsumerSpec]:
    """Translate ``capture_mode`` into a CaptureEngine consumer-spec list.

    Uses the filesystem consumer because it has ``reads_client_spec=True``,
    so per-request capture admission actually succeeds. The logging consumer
    is global-only and would silently reject the per-request spec needed by
    ``cap_on_active``, collapsing it into ``cap_on_idle``.
    """
    if capture_mode == "cap_off":
        return []
    if fs_root is None:
        raise ValueError("filesystem root required for cap_on_* modes")
    return [
        CaptureConsumerSpec(
            name="filesystem",
            params={"root": fs_root, "writer_threads": 4},
        )
    ]


CAPTURE_SPEC_PRESETS = ("minimal", "medium", "heavy")


def _build_request_capture(
    capture_spec: str,
    capture_layer: int,
    num_layers: int,
) -> RequestCapture:
    """Build the per-request capture opt-in for the filesystem consumer.

    Three presets that scale the per-step gather/dispatch work:

    - ``minimal``: 1 hook × 1 layer × ``last_prompt`` — 1 row per request.
      Floor case for "active capture overhead".
    - ``medium``: 1 hook × every 4th layer × ``all_generated`` —
      ``ceil(num_layers/4) * output_len`` rows per request. Stresses the
      manager's per-step gather + sink dispatch without saturating bandwidth.
    - ``heavy``: 2 hooks × all layers × ``all`` — ``2 * num_layers * (prompt_len + output_len)``
      rows per request. Stresses the gather kernel and the writer thread.
    """
    if capture_spec == "minimal":
        hooks = {"post_block": [capture_layer]}
        positions: str = "last_prompt"
    elif capture_spec == "medium":
        every_fourth = list(range(0, num_layers, 4))
        hooks = {"post_block": every_fourth}
        positions = "all_generated"
    elif capture_spec == "heavy":
        all_layers = list(range(num_layers))
        hooks = {"post_block": all_layers, "post_attn": all_layers}
        positions = "all"
    else:
        raise ValueError(
            f"unknown capture_spec preset {capture_spec!r}; "
            f"expected one of {CAPTURE_SPEC_PRESETS}"
        )
    return RequestCapture(
        consumer="filesystem",
        hooks=hooks,
        positions=positions,
        request_id="bench",
        tag="bench",
    )


def _build_requests(
    steering_mode: str,
    capture_mode: str,
    capture_spec: str,
    prompts: list[str],
    max_tokens: int,
    hidden_size: int,
    num_layers: int,
    capture_layer: int,
) -> list[GenerationRequest]:
    """Return the request batch for one (steering, capture) cell."""
    capture = None
    if capture_mode == "cap_on_active":
        capture = _build_request_capture(
            capture_spec=capture_spec,
            capture_layer=capture_layer,
            num_layers=num_layers,
        )

    def _req(prompt: str, steering: SteeringSpec | None) -> GenerationRequest:
        return GenerationRequest(
            prompt=prompt, max_tokens=max_tokens, steering=steering, capture=capture
        )

    if steering_mode in ("disabled", "enabled_idle"):
        return [_req(p, None) for p in prompts]

    if steering_mode == "per_request_1":
        spec = SteeringSpec.from_vector_dict(
            random_steering_vectors(
                hidden_size=hidden_size,
                num_layers=num_layers,
                hook_points=["post_block"],
                scale=0.1,
                seed=42,
            )
        )
        return [_req(p, spec) for p in prompts]

    if steering_mode == "per_request_4":
        diverse = random_steering_vectors_diverse(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_configs=4,
            hook_points=["post_block"],
            scale=0.1,
            base_seed=42,
        )
        specs = [SteeringSpec.from_vector_dict(d) for d in diverse]
        return [_req(p, specs[i % 4]) for i, p in enumerate(prompts)]

    raise ValueError(f"unknown steering mode: {steering_mode}")


def _run_cell(
    *,
    model: str,
    steering_mode: str,
    capture_mode: str,
    capture_spec: str,
    batch_size: int,
    prompt_len: int,
    max_tokens: int,
    warmup: int,
    iters: int,
    hidden_size: int,
    num_layers: int,
    capture_layer: int,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> dict:
    """Run a single (steering, capture, batch_size) cell. Picklable for spawn."""
    import tempfile

    enable_steering = steering_mode != "disabled"
    max_configs = 8 if steering_mode == "per_request_4" else 4

    fs_root: str | None = None
    fs_tmpdir = None
    if capture_mode != "cap_off":
        fs_tmpdir = tempfile.TemporaryDirectory(prefix="bench-capture-")
        fs_root = fs_tmpdir.name

    specs = _build_capture_consumers(capture_mode, capture_layer, fs_root)

    print(
        f"    [load] steering={enable_steering} max_configs={max_configs} "
        f"capture={capture_mode} prefix_cache={enable_prefix_caching}",
        flush=True,
    )
    engine = VllmSteeringEngine()
    engine.configure_capture(specs)
    engine.load(
        model,
        steering_config=SteeringConfig(
            enable_steering=enable_steering,
            max_steering_configs=max_configs,
            enable_prefix_caching=enable_prefix_caching,
        ),
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    prompts = _make_prompts(batch_size, prompt_len)
    requests = _build_requests(
        steering_mode=steering_mode,
        capture_mode=capture_mode,
        capture_spec=capture_spec,
        prompts=prompts,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        num_layers=num_layers,
        capture_layer=capture_layer,
    )

    print(f"    [measure] warmup={warmup} iters={iters}", flush=True)
    try:
        for _ in range(warmup):
            engine.generate(requests)
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.generate(requests)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        stats = compute_stats(samples)
        return stats.to_dict()
    except torch.cuda.OutOfMemoryError:
        return {"error": "OOM", "samples_ms": []}
    finally:
        engine.teardown()
        gc.collect()
        torch.cuda.empty_cache()
        if fs_tmpdir is not None:
            fs_tmpdir.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Latency of steering modes with/without capture-consumers enabled "
            "(measures whether installing a capture manager adds overhead to "
            "the steering hot path even when no request asks for a capture)."
        )
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument(
        "--steering-modes",
        default=",".join(STEERING_MODES),
        help=f"Comma-separated subset of {STEERING_MODES}",
    )
    parser.add_argument(
        "--capture-modes",
        default=",".join(CAPTURE_MODES),
        help=f"Comma-separated subset of {CAPTURE_MODES}",
    )
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=15,
        help="Layer index used by the minimal capture spec (clamped to num_layers-1).",
    )
    parser.add_argument(
        "--capture-spec",
        choices=CAPTURE_SPEC_PRESETS,
        default="minimal",
        help=(
            "Per-request capture spec preset for cap_on_active. "
            "minimal=1 hook×1 layer×last_prompt (default, ~1 row/req); "
            "medium=1 hook×every-4th-layer×all_generated; "
            "heavy=2 hooks×all layers×all positions."
        ),
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching to isolate per-step overhead.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help=(
            "Run all cells in this process (faster, but vLLM leaks weight memory "
            "across LLM constructions and may OOM after the first few cells)."
        ),
    )
    parser.add_argument(
        "--subprocess-timeout",
        type=float,
        default=900.0,
        help="Per-cell subprocess wall-clock budget in seconds.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=engine_names(),
        help="Engine adapter. This matrix exercises the vLLM-fork "
             "capture-consumer plugin system on the steering hot path; only "
             "--engine vllm is supported. For the engine-agnostic "
             "capture-overhead sweep use `python -m steering_bench run capture`.",
    )
    args = parser.parse_args()

    if args.engine != "vllm":
        parser.error(
            f"engine {args.engine!r} not supported: this script measures the "
            "vLLM-fork capture-consumer plugin overhead, which has no "
            "engine-agnostic equivalent. Use --engine vllm, or "
            "`python -m steering_bench run capture` for the portable subset."
        )

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    model_cfg = get_model_config(args.model)
    hidden_size = model_cfg.hidden_size
    num_layers = model_cfg.num_layers
    capture_layer = min(args.capture_layer, num_layers - 1)
    engine_identity = VllmSteeringEngine().identity()

    steering_modes = [m for m in args.steering_modes.split(",") if m]
    capture_modes = [m for m in args.capture_modes.split(",") if m]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    for m in steering_modes:
        if m not in STEERING_MODES:
            print(f"ERROR: unknown steering mode {m!r}")
            sys.exit(2)
    for m in capture_modes:
        if m not in CAPTURE_MODES:
            print(f"ERROR: unknown capture mode {m!r}")
            sys.exit(2)

    print(f"steering_with_capture: {args.model}")
    print(f"  steering_modes = {steering_modes}")
    print(f"  capture_modes  = {capture_modes}")
    print(f"  batch_sizes    = {batch_sizes}")
    print(f"  capture_layer  = {capture_layer}")
    print(f"  prefix_cache   = {not args.disable_prefix_cache}")
    print()

    # Baselines for overhead %: keyed by (steering_mode, batch_size); we
    # compare each capture_mode at fixed (steering, batch) against cap_off.
    # If cap_off isn't in the requested capture_modes the overhead column
    # stays None.
    baselines: dict[tuple[str, int], float] = {}
    all_results: list[dict] = []

    for steering_mode in steering_modes:
        for capture_mode in capture_modes:
            print(f"\n--- steering={steering_mode}  capture={capture_mode} ---")
            for batch_size in batch_sizes:
                cell_kwargs = dict(
                    model=args.model,
                    steering_mode=steering_mode,
                    capture_mode=capture_mode,
                    capture_spec=args.capture_spec,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    max_tokens=args.max_tokens,
                    warmup=args.warmup,
                    iters=args.iters,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    capture_layer=capture_layer,
                    enable_prefix_caching=not args.disable_prefix_cache,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                )

                print(f"  batch_size={batch_size}")
                if args.no_subprocess:
                    result = _run_cell(**cell_kwargs)
                else:
                    result = run_in_subprocess(
                        target=_run_cell,
                        kwargs=cell_kwargs,
                        timeout_s=args.subprocess_timeout,
                    )

                if "error" in result and "mean_ms" not in result:
                    print(f"    {result['error']}")
                    overhead_pct = None
                else:
                    mean = result["mean_ms"]
                    p90 = result.get("p90_ms", float("nan"))
                    print(f"    mean={mean:.1f}ms p90={p90:.1f}ms")

                    if capture_mode == "cap_off":
                        baselines[(steering_mode, batch_size)] = mean

                    overhead_pct = None
                    base = baselines.get((steering_mode, batch_size))
                    if base is not None and base > 0:
                        overhead_pct = (mean - base) / base * 100.0
                        if capture_mode == "cap_off":
                            print(f"    (baseline for steering={steering_mode})")
                        else:
                            print(
                                f"    overhead vs cap_off: {overhead_pct:+.2f}%"
                            )

                params = {
                    "model": args.model,
                    "steering_mode": steering_mode,
                    "capture_mode": capture_mode,
                    "capture_spec": args.capture_spec,
                    "batch_size": batch_size,
                    "prompt_len": args.prompt_len,
                    "max_tokens": args.max_tokens,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "capture_layer": capture_layer,
                    "prefix_caching": not args.disable_prefix_cache,
                }
                results_dict: dict = {
                    "latency_ms": {
                        k: v for k, v in result.items() if k != "samples_ms"
                    },
                }
                if overhead_pct is not None:
                    results_dict["overhead_pct_vs_cap_off"] = overhead_pct

                write_result(
                    benchmark="vllm.steering_with_capture",
                    parameters=params,
                    results=results_dict,
                    output_dir=args.output_dir,
                    tag=args.tag,
                    raw_samples_ms=result.get("samples_ms"),
                    engine=engine_identity,
                )
                all_results.append({
                    "steering_mode": steering_mode,
                    "capture_mode": capture_mode,
                    "batch_size": batch_size,
                    "mean_ms": result.get("mean_ms"),
                    "p90_ms": result.get("p90_ms"),
                    "overhead_pct": overhead_pct,
                    "error": result.get("error"),
                })

    # Summary
    print("\n" + "=" * 96)
    print(f"  Steering ⨯ Capture latency: {args.model}")
    print("=" * 96)
    header = (
        f"{'steering':<14} {'capture':<14} {'batch':>6} "
        f"{'mean_ms':>10} {'p90_ms':>10} {'overhead':>10}"
    )
    print(header)
    print("-" * 96)
    for r in all_results:
        if r["error"] and r["mean_ms"] is None:
            print(
                f"{r['steering_mode']:<14} {r['capture_mode']:<14} "
                f"{r['batch_size']:>6} {'ERROR':>10}"
            )
            continue
        overhead = (
            f"{r['overhead_pct']:+.2f}%"
            if r["overhead_pct"] is not None
            else "baseline"
        )
        print(
            f"{r['steering_mode']:<14} {r['capture_mode']:<14} "
            f"{r['batch_size']:>6} {r['mean_ms']:>10.1f} "
            f"{r['p90_ms']:>10.1f} {overhead:>10}"
        )
    print("=" * 96)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
