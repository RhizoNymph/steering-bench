"""Engine-agnostic steering **mode** catalog.

The offline steering modes that used to be ``--engine vllm``-guarded in
``scripts/bench_latency.py`` / ``scripts/bench_throughput.py``, expressed as
data over the engine seam so any engine can drive them:

- ``disabled`` -- no steering; load with ``enable_steering=False``.  Baseline.
- ``enabled_idle`` -- steering enabled at load, no per-request vectors.  Measures
  the fixed steering-on overhead in the absence of actual vectors.
- ``inline_shared`` -- one ``SteeringSpec`` on every request.
- ``inline_unique`` -- a fresh ``SteeringSpec`` per request (defeats reuse
  amortization).
- ``named_shared`` -- register one module once and have every request reference
  it via ``NamedModuleRef(name, scale)``.  On an engine without
  ``named_modules`` this **degrades to inline_shared** (same vectors, delivered
  inline) with a printed notice, so the cross-engine comparison still produces
  data.
- ``per_request_N`` -- cycle ``N`` distinct specs across the batch.

Each mode maps to how it builds requests, whether it needs a registered module,
and the load-time :class:`SteeringConfig` (``enable_steering`` +
``max_steering_configs`` auto-sizing + prefix caching).
"""

from __future__ import annotations

import argparse
from typing import Any, ClassVar

from steering_bench.engine.base import SteeringConfig
from steering_bench.engine.spec import GenerationRequest, NamedModuleRef, SteeringSpec
from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.workload import (
    diverse_steering_specs,
    make_prompt,
    steering_spec_for,
)

# Fixed name the ``named_shared`` mode registers and every request references.
NAMED_MODULE = "bench_named_shared"
NAMED_SCALE = 1.0

# The named modes catalog (excludes the open-ended ``per_request_N`` family,
# which is validated dynamically).
BASE_MODES: tuple[str, ...] = (
    "disabled",
    "enabled_idle",
    "inline_shared",
    "inline_unique",
    "named_shared",
)


class ModeError(ValueError):
    """Raised when a mode string is not a recognized steering mode."""


def parse_mode(mode: str) -> tuple[str, int | None]:
    """Split a mode string into ``(base, n)``.

    ``per_request_4`` -> ``("per_request", 4)``; every other recognized mode
    -> ``(mode, None)``.  Raises :class:`ModeError` on an unknown mode.
    """
    if mode in BASE_MODES:
        return mode, None
    if mode.startswith("per_request_"):
        suffix = mode[len("per_request_") :]
        try:
            n = int(suffix)
        except ValueError:
            raise ModeError(f"invalid per_request mode {mode!r}") from None
        if n < 1:
            raise ModeError(f"per_request_N requires N >= 1, got {n}")
        return "per_request", n
    raise ModeError(
        f"unknown mode {mode!r}; known: {', '.join(BASE_MODES)}, per_request_N"
    )


def mode_enable_steering(mode: str) -> bool:
    """Whether the steering subsystem is enabled at load for ``mode``."""
    base, _ = parse_mode(mode)
    return base != "disabled"


def mode_needs_named_module(mode: str) -> bool:
    """Whether ``mode`` registers and references a named steering module."""
    base, _ = parse_mode(mode)
    return base == "named_shared"


def max_steering_configs_for(mode: str, batch_size: int) -> int:
    """Auto-size the worker steering-config table for ``mode`` at ``batch_size``.

    Mirrors the per-mode sizing in the legacy scripts: ``inline_unique`` needs
    room for every distinct per-iter spec plus warmup/timed headroom; the
    ``per_request_N`` family needs both inline-packed and auto-promoted named
    slots (``max(4N, 16)``); everything else uses the default of ``4``.
    """
    base, n = parse_mode(mode)
    if base == "inline_unique":
        return max(64, batch_size * 2)
    if base == "per_request":
        assert n is not None
        return max(n * 4, 16)
    return 4


def steering_config_for(
    mode: str, batch_size: int, *, enable_prefix_caching: bool = True
) -> SteeringConfig:
    """Build the load-time :class:`SteeringConfig` for ``mode`` at ``batch_size``."""
    return SteeringConfig(
        enable_steering=mode_enable_steering(mode),
        max_steering_configs=max_steering_configs_for(mode, batch_size),
        enable_prefix_caching=enable_prefix_caching,
    )


def module_spec_for(model: str, layer: int, hook: str) -> SteeringSpec:
    """The spec registered by ``named_shared`` (and its inline fallback).

    Identical to the ``inline_shared`` spec so ``named_shared`` references the
    same vectors, just delivered by name rather than inline.
    """
    return steering_spec_for(model, layer, hook)


def build_mode_requests(
    mode: str,
    *,
    model: str,
    layer: int,
    hook: str,
    max_tokens: int,
    batch_size: int,
    prompt_len: int = 64,
    named_module: str = NAMED_MODULE,
    named_fallback: bool = False,
) -> list[GenerationRequest]:
    """Build the request batch for ``mode``.

    ``named_fallback`` (set when the target engine lacks ``named_modules``)
    turns ``named_shared`` into an inline-shared batch using the same vectors.
    """
    base, n = parse_mode(mode)
    prompt = make_prompt(prompt_len)

    def req(steering: SteeringSpec | NamedModuleRef | None) -> GenerationRequest:
        return GenerationRequest(prompt=prompt, max_tokens=max_tokens, steering=steering)

    match base:
        case "disabled" | "enabled_idle":
            return [req(None) for _ in range(batch_size)]
        case "inline_shared":
            spec = steering_spec_for(model, layer, hook)
            return [req(spec) for _ in range(batch_size)]
        case "inline_unique":
            specs = diverse_steering_specs(model, layer, hook, batch_size)
            return [req(spec) for spec in specs]
        case "named_shared":
            if named_fallback:
                spec = module_spec_for(model, layer, hook)
                return [req(spec) for _ in range(batch_size)]
            ref = NamedModuleRef(named_module, NAMED_SCALE)
            return [req(ref) for _ in range(batch_size)]
        case "per_request":
            assert n is not None
            specs = diverse_steering_specs(model, layer, hook, n)
            return [req(specs[i % n]) for i in range(batch_size)]
    raise ModeError(f"unhandled mode base {base!r}")  # pragma: no cover


DEFAULT_MODE = "inline_shared"
DEFAULT_PROMPT_LEN = 64


class ModeBenchmark(Benchmark):
    """Base for mode-driven, engine-agnostic offline steering benchmarks.

    Subclasses pick the batch-flag surface (``--batch-size`` for latency,
    ``--num-prompts`` for throughput) via the class attributes below; the mode
    catalog (:func:`build_mode_requests`, :func:`steering_config_for`) drives
    request construction, load-time config, and named-module registration.
    ``named_shared`` degrades to inline-shared on engines without
    ``named_modules`` (a printed notice + ``named_shared_fallback`` in params).
    """

    #: CLI flag / parsed-attr / default for the batch-size axis.
    batch_flag: ClassVar[str] = "--batch-size"
    batch_dest: ClassVar[str] = "batch_size"
    default_batch: ClassVar[int] = 1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mode",
            default=DEFAULT_MODE,
            help=(
                "Steering mode: disabled, enabled_idle, inline_shared, "
                "inline_unique, named_shared, or per_request_N."
            ),
        )
        parser.add_argument(
            cls.batch_flag,
            type=int,
            default=cls.default_batch,
            help="Requests generated per measured iteration.",
        )
        parser.add_argument(
            "--prompt-len",
            type=int,
            default=DEFAULT_PROMPT_LEN,
            help="Approximate prompt length in tokens.",
        )
        parser.add_argument(
            "--disable-prefix-cache",
            action="store_true",
            help="Disable prefix caching (vLLM only; no-op on other engines).",
        )

    @classmethod
    def options_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        # Validate the mode early so a bad --mode fails before load.
        parse_mode(args.mode)
        return {
            "mode": args.mode,
            "batch_size": getattr(args, cls.batch_dest),
            "prompt_len": args.prompt_len,
            "enable_prefix_caching": not args.disable_prefix_cache,
        }

    # -- convenience accessors ----------------------------------------------

    @property
    def mode(self) -> str:
        return str(self.options.get("mode", DEFAULT_MODE))

    @property
    def batch_size(self) -> int:
        return int(self.options.get("batch_size", self.default_batch))

    @property
    def prompt_len(self) -> int:
        return int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN))

    @property
    def enable_prefix_caching(self) -> bool:
        return bool(self.options.get("enable_prefix_caching", True))

    def uses_named_module(self) -> bool:
        return mode_needs_named_module(self.mode)

    def named_fallback(self) -> bool:
        """True when ``named_shared`` must degrade to inline for this engine."""
        return self.uses_named_module() and not self.engine.capabilities.named_modules

    # -- Benchmark hooks -----------------------------------------------------

    def build_requests(self) -> list[GenerationRequest]:
        return build_mode_requests(
            self.mode,
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
            max_tokens=self.config.max_tokens,
            batch_size=self.batch_size,
            prompt_len=self.prompt_len,
            named_fallback=self.named_fallback(),
        )

    def steering_config(self) -> SteeringConfig:
        return steering_config_for(
            self.mode,
            self.batch_size,
            enable_prefix_caching=self.enable_prefix_caching,
        )

    def after_load(self) -> None:
        if not self.uses_named_module():
            return
        if self.named_fallback():
            print(
                f"    note: engine {self.engine.name!r} lacks named modules; "
                f"mode 'named_shared' degraded to inline-shared (same vectors)."
            )
            return
        self.engine.register_module(
            NAMED_MODULE,
            module_spec_for(self.config.model, self.config.layer, self.config.hook),
            replace=True,
        )

    def parameters(self) -> dict[str, Any]:
        params = super().parameters()
        params["mode"] = self.mode
        params[self.batch_dest] = self.batch_size
        params["prompt_len"] = self.prompt_len
        params["prefix_caching"] = self.enable_prefix_caching
        params["max_steering_configs"] = max_steering_configs_for(
            self.mode, self.batch_size
        )
        if self.named_fallback():
            params["named_shared_fallback"] = True
        return params
