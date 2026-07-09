"""Online-serving benchmark driven through the :class:`ServingEngine` seam.

The engine-agnostic successor to ``scripts/bench_serving.py``: launches an API
server, drives streaming completions per steering *mode*, and records the
per-token latency profile (TTFT / TPOT / ITL / E2EL). Unlike the synchronous
:class:`~steering_bench.harness.benchmark.Benchmark`, serving needs a PARALLEL
async path, so :class:`ServingBenchmark` owns its own lifecycle (server
start/stop is synchronous; each server phase's measurement runs under a single
``asyncio.run``) rather than subclassing the sync measure-loop.

Modes (mirrors the legacy script):
  * ``disabled``           -- server without ``--enable-steering``; baseline.
  * ``enabled_idle``       -- steering enabled, no vectors on requests.
  * ``all_steered_shared`` -- every request carries the same inline spec.
  * ``named_shared``       -- register one module, reference it by name per request.
  * ``per_request_n4`` / ``per_request_n16`` -- cycle N distinct inline specs.

``disabled`` needs its own server (steering off); the steered modes share one
``--enable-steering`` server. Payload packing, registration, and the timing dump
are owned by the adapter -- this benchmark only speaks typed
:class:`~steering_bench.engine.spec.SteeringSpec` / ``NamedModuleRef``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

from steering_bench.engine.serving import (
    ServingConfig,
    ServingEngine,
    summarize_results,
)
from steering_bench.engine.spec import (
    GenerationRequest,
    NamedModuleRef,
    SteeringSpec,
)
from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.harness.models import get_model_config
from steering_bench.output import write_result
from steering_bench.vectors import (
    random_steering_vectors,
    random_steering_vectors_diverse,
)

SERVING_MODES: tuple[str, ...] = (
    "disabled",
    "enabled_idle",
    "all_steered_shared",
    "named_shared",
    "per_request_n4",
    "per_request_n16",
)
DEFAULT_MODES = ",".join(SERVING_MODES)
NAMED_BENCH_MODULE = "bench_named_shared"


class ServingModeError(ValueError):
    """Raised when a serving mode string is unrecognized."""


def per_request_count(mode: str) -> int | None:
    """N for a ``per_request_nK`` mode, else ``None``. Raises on garbage suffix."""
    if not mode.startswith("per_request_n"):
        return None
    suffix = mode[len("per_request_n") :]
    try:
        n = int(suffix)
    except ValueError:
        raise ServingModeError(f"invalid per_request mode {mode!r}") from None
    if n < 1:
        raise ServingModeError(f"per_request_nK requires N >= 1, got {n}")
    return n


def validate_modes(modes: list[str]) -> None:
    """Raise :class:`ServingModeError` if any mode is unrecognized."""
    for mode in modes:
        if mode in SERVING_MODES:
            continue
        if per_request_count(mode) is not None:
            continue
        raise ServingModeError(
            f"unknown mode {mode!r}; known: {', '.join(SERVING_MODES)}"
        )


def _per_row_scales(num_layers: int) -> tuple[float, ...]:
    """Deterministic per-row scales (all != 1.0) exercising the multiply path."""
    return tuple(round(0.5 + 0.05 * i, 4) for i in range(num_layers))


def shared_spec_for(
    model: str, hook: str, *, seed: int = 42, with_scales: bool = False
) -> SteeringSpec:
    """The all-layers shared inline spec (optionally carrying per-row scales)."""
    cfg = get_model_config(model)
    vectors = random_steering_vectors(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        hook_points=[hook],
        scale=0.1,
        seed=seed,
    )
    spec = SteeringSpec.from_vector_dict(vectors)
    if with_scales:
        return spec.with_scales(_per_row_scales(spec.num_vectors()))
    return spec


def diverse_specs_for(
    model: str,
    hook: str,
    num_configs: int,
    *,
    base_seed: int,
    with_scales: bool = False,
) -> list[SteeringSpec]:
    """``num_configs`` distinct all-layers inline specs."""
    cfg = get_model_config(model)
    diverse = random_steering_vectors_diverse(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_configs=num_configs,
        hook_points=[hook],
        scale=0.1,
        base_seed=base_seed,
    )
    specs = [SteeringSpec.from_vector_dict(v) for v in diverse]
    if with_scales:
        return [s.with_scales(_per_row_scales(s.num_vectors())) for s in specs]
    return specs


def distinct_configs_for_mode(mode: str, diverse: list[SteeringSpec]) -> int:
    """Warmup floor: how many distinct configs a mode first-touches."""
    if mode in ("disabled", "enabled_idle"):
        return 0
    if mode in ("named_shared", "all_steered_shared"):
        return 1
    return len(diverse)


def build_serving_requests(
    mode: str,
    prompts: list[str],
    max_tokens: int,
    *,
    shared: SteeringSpec,
    diverse: list[SteeringSpec],
    named_module: str = NAMED_BENCH_MODULE,
) -> list[GenerationRequest]:
    """Build the per-request batch for ``mode`` as typed requests.

    Steering encoding stays in the adapter: this only chooses the ``SteeringSpec``
    / ``NamedModuleRef`` / ``None`` each request carries.
    """
    n = len(prompts)

    def req(prompt: str, steering: Any) -> GenerationRequest:
        return GenerationRequest(prompt=prompt, max_tokens=max_tokens, steering=steering)

    if mode in ("disabled", "enabled_idle"):
        return [req(p, None) for p in prompts]
    if mode == "all_steered_shared":
        return [req(p, shared) for p in prompts]
    if mode == "named_shared":
        ref = NamedModuleRef(named_module)
        return [req(p, ref) for p in prompts]
    k = len(diverse)
    if k == 0:
        raise ServingModeError(f"mode {mode!r} needs diverse specs")
    return [req(prompts[i], diverse[i % k]) for i in range(n)]


# -- prompt loaders (ported from the script) ---------------------------------


def make_synthetic_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts


def load_sharegpt(
    path: Path, num_prompts: int, min_words: int, max_words: int
) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    prompts: list[str] = []
    for entry in data:
        conv = entry.get("conversations") or []
        if not conv or conv[0].get("from") != "human":
            continue
        text = conv[0].get("value", "").strip()
        if min_words <= len(text.split()) <= max_words:
            prompts.append(text)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"ShareGPT only yielded {len(prompts)} prompts matching "
            f"{min_words}-{max_words} words; needed {num_prompts}"
        )
    return prompts


def _print_summary(mode: str, n: int, s: dict[str, Any]) -> None:
    if s.get("num_ok", 0) == 0:
        print(f"  {mode}: ALL FAILED ({s.get('num_err', 0)} errors)")
        return

    def m(section: str) -> str:
        return f"{s.get(section, {}).get('median_ms', float('nan')):.1f}"

    print(
        f"  {mode:<22} n={n} TTFT={m('ttft_ms')}ms TPOT={m('tpot_ms')}ms "
        f"ITL={m('itl_ms')}ms E2EL={m('e2el_ms')}ms "
        f"throughput={s.get('offline_output_tps', 0):.0f}tok/s"
    )


class ServingBenchmark:
    """Online-serving benchmark orchestrator (own async lifecycle).

    Not a :class:`Benchmark` subclass: serving's streaming, multi-server-phase
    lifecycle does not fit the synchronous warmup/measure loop. Constructed with
    a :class:`ServingEngine` and the shared :class:`BenchmarkConfig`; serving
    flags arrive via ``options``.
    """

    benchmark_name: ClassVar[str] = "serving"
    is_comparison: ClassVar[bool] = False

    def __init__(
        self, engine: ServingEngine, config: BenchmarkConfig, **options: Any
    ) -> None:
        self.engine = engine
        self.config = config
        self.options = options

    # -- CLI surface ---------------------------------------------------------

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--num-prompts", type=int, default=64)
        parser.add_argument("--concurrency", type=int, default=16)
        parser.add_argument("--prompt-len", type=int, default=256)
        parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
        parser.add_argument("--max-steering-configs", type=int, default=16)
        parser.add_argument("--startup-timeout", type=float, default=240.0)
        parser.add_argument(
            "--warmup-requests",
            type=int,
            default=None,
            help="Discarded warmup requests per mode (default: --concurrency; 0 disables).",
        )
        parser.add_argument("--warmup-max-tokens", type=int, default=8)
        parser.add_argument("--warmup-drain-seconds", type=float, default=0.5)
        parser.add_argument(
            "--packed-with-scales",
            action="store_true",
            help="Attach deterministic per-row scales to inline specs.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Pass --enforce-eager to the server (disables CUDA graphs).",
        )
        parser.add_argument("--sharegpt-path", default=None)
        parser.add_argument(
            "--modes",
            default=DEFAULT_MODES,
            help=f"Comma-separated subset of: {', '.join(SERVING_MODES)}.",
        )

    @staticmethod
    def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        validate_modes(modes)
        return {
            "modes": modes,
            "port": args.port,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "prompt_len": args.prompt_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_steering_configs": args.max_steering_configs,
            "startup_timeout": args.startup_timeout,
            "warmup_requests": args.warmup_requests,
            "warmup_max_tokens": args.warmup_max_tokens,
            "warmup_drain_seconds": args.warmup_drain_seconds,
            "packed_with_scales": args.packed_with_scales,
            "enforce_eager": args.enforce_eager,
            "sharegpt_path": args.sharegpt_path,
        }

    # -- accessors -----------------------------------------------------------

    def _opt(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def _warmup_requests(self) -> int:
        wr = self._opt("warmup_requests")
        return int(wr) if wr is not None else int(self._opt("concurrency", 16))

    # -- prompts / specs -----------------------------------------------------

    def _build_prompts(self) -> tuple[list[str], str]:
        sharegpt = self._opt("sharegpt_path")
        num = int(self._opt("num_prompts", 64))
        if sharegpt:
            prompts = load_sharegpt(Path(sharegpt), num, 32, 512)
            return prompts, "sharegpt"
        return make_synthetic_prompts(num, int(self._opt("prompt_len", 256))), "synthetic"

    # -- parameters ----------------------------------------------------------

    def _parameters(self, workload: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "engine": self.engine.name,
            "workload": workload,
            "num_prompts": int(self._opt("num_prompts", 64)),
            "concurrency": int(self._opt("concurrency", 16)),
            "max_tokens": self.config.max_tokens,
            "prompt_len": int(self._opt("prompt_len", 256)) if workload == "synthetic" else None,
            "max_model_len": int(self._opt("max_model_len", 4096)),
            "sharegpt_path": self._opt("sharegpt_path"),
            "warmup_requests": self._warmup_requests(),
            "warmup_max_tokens": int(self._opt("warmup_max_tokens", 8)),
            "warmup_drain_seconds": float(self._opt("warmup_drain_seconds", 0.5)),
            "packed_with_scales": bool(self._opt("packed_with_scales", False)),
            "enforce_eager": bool(self._opt("enforce_eager", False)),
        }

    def _server_config(self, *, enable_steering: bool, dev_mode: bool) -> ServingConfig:
        return ServingConfig(
            enable_steering=enable_steering,
            max_steering_configs=int(self._opt("max_steering_configs", 16)),
            dev_mode=dev_mode,
            timing=True,
            enforce_eager=bool(self._opt("enforce_eager", False)),
            max_model_len=int(self._opt("max_model_len", 4096)),
            gpu_memory_utilization=float(self._opt("gpu_memory_utilization", 0.9)),
            port=int(self._opt("port", 8765)),
            startup_timeout=float(self._opt("startup_timeout", 240.0)),
        )

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        prompts, workload = self._build_prompts()
        modes: list[str] = list(self._opt("modes", list(SERVING_MODES)))
        params_base = self._parameters(workload)
        with_scales = bool(self._opt("packed_with_scales", False))
        hook = self.config.hook

        shared = shared_spec_for(self.config.model, hook, with_scales=with_scales)
        diverse_cache: dict[int, list[SteeringSpec]] = {}

        def diverse_for(n: int) -> list[SteeringSpec]:
            if n not in diverse_cache:
                diverse_cache[n] = diverse_specs_for(
                    self.config.model, hook, n, base_seed=100 + n, with_scales=with_scales
                )
            return diverse_cache[n]

        summaries: dict[str, dict[str, Any]] = {}

        # Phase 1: disabled needs its own (steering-off) server.
        if "disabled" in modes:
            cfg = self._server_config(enable_steering=False, dev_mode=False)
            self.engine.start_server(self.config.model, config=cfg)
            try:
                print("\n[phase] disabled")
                summaries.update(
                    asyncio.run(
                        self._run_modes(
                            ["disabled"], prompts, params_base, shared, diverse_for,
                            {**params_base, "enable_steering": False},
                        )
                    )
                )
            finally:
                self.engine.stop_server()

        # Phase 2: shared enable-steering server for every steered mode.
        steered = [m for m in modes if m != "disabled"]
        if steered:
            needs_dev = "named_shared" in steered
            cfg = self._server_config(enable_steering=True, dev_mode=needs_dev)
            self.engine.start_server(self.config.model, config=cfg)
            try:
                print("\n[phase] enable_steering")
                summaries.update(
                    asyncio.run(
                        self._run_modes(
                            steered, prompts, params_base, shared, diverse_for,
                            {
                                **params_base,
                                "enable_steering": True,
                                "max_steering_configs": int(self._opt("max_steering_configs", 16)),
                            },
                        )
                    )
                )
            finally:
                self.engine.stop_server()

        print(f"\nResults written to {self.config.output_dir}")
        return {"benchmark": self.benchmark_name, "summaries": summaries}

    async def _run_modes(
        self,
        modes: list[str],
        prompts: list[str],
        params_base: dict[str, Any],
        shared: SteeringSpec,
        diverse_for: Any,
        params_extra: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        concurrency = int(self._opt("concurrency", 16))
        max_tokens = self.config.max_tokens
        warmup_reqs = self._warmup_requests()
        warmup_max_tokens = int(self._opt("warmup_max_tokens", 8))
        drain = float(self._opt("warmup_drain_seconds", 0.5))

        if "named_shared" in modes:
            await self.engine.register_named_module(NAMED_BENCH_MODULE, shared)

        out: dict[str, dict[str, Any]] = {}
        for mode in modes:
            n = per_request_count(mode)
            diverse = diverse_for(n) if n is not None else []
            requests = build_serving_requests(
                mode, prompts, max_tokens, shared=shared, diverse=diverse
            )
            mode_warmup = max(warmup_reqs, distinct_configs_for_mode(mode, diverse))
            if mode_warmup > 0:
                w = min(mode_warmup, len(requests))
                print(f"  {mode}: warmup ({w} reqs, max_tokens={warmup_max_tokens})")
                warm_reqs = [
                    GenerationRequest(
                        prompt=r.prompt, max_tokens=warmup_max_tokens, steering=r.steering
                    )
                    for r in requests[:w]
                ]
                await self.engine.warmup(
                    warm_reqs, concurrency=concurrency, drain_seconds=drain
                )
                await self.engine.dump_and_reset_timings(mode, quiet=True)

            results = await self.engine.run_workload(requests, concurrency)
            summary = summarize_results(results)
            _print_summary(mode, len(requests), summary)
            write_result(
                benchmark=f"{self.engine.name}.serving",
                parameters={**params_extra, "mode": mode},
                results=summary,
                output_dir=self.config.output_dir,
                tag=self.config.tag,
                engine=self.engine.identity(),
            )
            await self.engine.dump_and_reset_timings(mode)
            out[mode] = summary
        return out
