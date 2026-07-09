"""Cross-library patch-sweep (causal-tracing) comparison through the seam.

The engine-agnostic successor to ``scripts/bench_patching_external.py``: runs the
identical denoising study -- patch the clean run's residual into the corrupt run
at every ``(layer, position)`` cell, grade the answer token -- through the
:class:`~steering_bench.engine.patch_sweep.PatchSweepEngine` adapters and compares
their throughput and argmax cell.

Like :class:`~steering_bench.harness.benchmarks.serving.ServingBenchmark`, this is
NOT a :class:`~steering_bench.harness.benchmark.Benchmark` subclass: patch sweep's
result shape (``cells / wall_s / cells_per_s / argmax``) is a distinct axis, not
the synchronous warmup/measure latency loop, and a single run compares *two*
engine types (TransformerLens in-process + vLLM over HTTP) rather than one
``--engine``. It owns its lifecycle and picks engines from ``--variants``.

Variants:
  * ``tl_naive``   -- TransformerLens, one forward per cell.
  * ``tl_batched`` -- TransformerLens, one forward per layer (position-batched).
  * ``vllm_sweep`` -- vLLM one-call ``POST /v1/patch_sweep`` (needs ``--base-url``).

Each engine's runs are written to their own result file stamped with the engine
identity block; the cross-tool argmax-agreement check and the cells/s summary are
printed (the comparison output the legacy script produced).
"""

from __future__ import annotations

import argparse
from typing import Any, ClassVar

from steering_bench.engine.patch_sweep import (
    PatchSweepEngine,
    PatchSweepError,
    PatchSweepRequest,
    PatchSweepResult,
    discover_patch_sweep,
)
from steering_bench.harness.benchmark import BenchmarkConfig
from steering_bench.output import write_result

_FILLER = (
    "In this geography quiz, we will answer questions about European "
    "countries, their capitals, rivers, and mountains. We will discuss "
    "history, culture, food, languages, and famous landmarks of each "
    "nation in turn, considering both ancient origins and modern life. "
)
PROMPT_PAIRS: dict[str, tuple[str, str]] = {
    "short": (
        "The capital of France is",
        "The capital of Germany is",
    ),
    "long": (
        "In this geography quiz, we will answer questions about European "
        "countries, their capitals, rivers, and mountains. Read each "
        "question carefully and answer with exactly one word. Question one. "
        "The capital of France is",
        "In this geography quiz, we will answer questions about European "
        "countries, their capitals, rivers, and mountains. Read each "
        "question carefully and answer with exactly one word. Question one. "
        "The capital of Germany is",
    ),
    "xl": (
        _FILLER * 4 + "Question one. The capital of France is",
        _FILLER * 4 + "Question one. The capital of Germany is",
    ),
}
ANSWER = " Paris"

ALL_VARIANTS: tuple[str, ...] = ("tl_naive", "tl_batched", "vllm_sweep")
DEFAULT_VARIANTS = ",".join(ALL_VARIANTS)
DEFAULT_PROMPTS = "short,long,xl"


class PatchSweepModeError(ValueError):
    """Raised when a variant or prompt name is unrecognized."""


def parse_variants(variants: list[str]) -> tuple[list[str], bool]:
    """Split a variant list into TransformerLens variants + a vLLM flag.

    Returns ``(tl_variants, want_vllm)`` where ``tl_variants`` are the bare
    mechanism names (``naive`` / ``batched``) stripped of the ``tl_`` prefix.
    Raises on any unknown variant.
    """
    tl_variants: list[str] = []
    want_vllm = False
    for v in variants:
        if v == "vllm_sweep":
            want_vllm = True
        elif v.startswith("tl_"):
            tl_variants.append(v.removeprefix("tl_"))
        else:
            raise PatchSweepModeError(
                f"unknown variant {v!r}; known: {', '.join(ALL_VARIANTS)}"
            )
    return tl_variants, want_vllm


def validate_prompts(prompts: list[str]) -> None:
    """Raise :class:`PatchSweepModeError` if any prompt name is unknown."""
    for p in prompts:
        if p not in PROMPT_PAIRS:
            raise PatchSweepModeError(
                f"unknown prompt {p!r}; known: {', '.join(PROMPT_PAIRS)}"
            )


def _print_run(label: str, r: PatchSweepResult) -> None:
    print(
        f"  [{label}] {r.cells} cells in {r.wall_s}s "
        f"({r.cells_per_s} cells/s), "
        f"argmax L{r.argmax.layer}@{r.argmax.position}"
    )


class PatchSweepBenchmark:
    """Cross-library patch-sweep comparison orchestrator (own lifecycle).

    Not a :class:`Benchmark` subclass: patch sweep is a distinct axis whose one
    run spans two engine types. Constructed with the shared
    :class:`BenchmarkConfig` (``model`` / ``output_dir`` / ``tag`` are the used
    fields) plus patch-sweep ``options``; it discovers and drives the adapters
    itself.
    """

    benchmark_name: ClassVar[str] = "patch-sweep"

    def __init__(self, config: BenchmarkConfig, **options: Any) -> None:
        self.config = config
        self.options = options

    # -- CLI surface ---------------------------------------------------------

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num-layers", type=int, default=28)
        parser.add_argument("--dtype", default="bfloat16")
        parser.add_argument("--base-url", default="http://localhost:8000/v1")
        parser.add_argument(
            "--variants",
            default=DEFAULT_VARIANTS,
            help=f"Comma-separated subset of: {', '.join(ALL_VARIANTS)}.",
        )
        parser.add_argument(
            "--prompts",
            default=DEFAULT_PROMPTS,
            help=f"Comma-separated subset of: {', '.join(PROMPT_PAIRS)}.",
        )
        parser.add_argument(
            "--reps", type=int, default=3, help="vllm_sweep repetitions per prompt."
        )
        parser.add_argument(
            "--logits-chunk-budget",
            type=int,
            default=4096,
            help="TransformerLens batched logits materialization budget.",
        )

    @staticmethod
    def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        parse_variants(variants)  # validate early
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        validate_prompts(prompts)
        return {
            "variants": variants,
            "prompts": prompts,
            "num_layers": args.num_layers,
            "dtype": args.dtype,
            "base_url": args.base_url,
            "reps": args.reps,
            "logits_chunk_budget": args.logits_chunk_budget,
        }

    # -- accessors -----------------------------------------------------------

    def _opt(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def _parameters(self) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "num_layers": int(self._opt("num_layers", 28)),
            "dtype": self._opt("dtype", "bfloat16"),
            "prompts": list(self._opt("prompts", list(PROMPT_PAIRS))),
            "variants": list(self._opt("variants", list(ALL_VARIANTS))),
            "answer": ANSWER,
            "base_url": self._opt("base_url"),
            "reps": int(self._opt("reps", 3)),
        }

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        variants: list[str] = list(self._opt("variants", list(ALL_VARIANTS)))
        prompts: list[str] = list(self._opt("prompts", list(PROMPT_PAIRS)))
        tl_variants, want_vllm = parse_variants(variants)
        params_base = self._parameters()

        wanted = ["tl"] if tl_variants else []
        if want_vllm:
            wanted.append("vllm")

        print("Discovering patch-sweep engines:")
        available = {cls.name: cls for cls in discover_patch_sweep(filter_names=wanted)}

        all_runs: list[dict[str, Any]] = []

        if tl_variants:
            if "tl" in available:
                all_runs += self._run_tl(
                    available["tl"](), tl_variants, prompts, params_base
                )
            # else: discover_patch_sweep already printed the skip reason.

        if want_vllm and "vllm" in available:
            all_runs += self._run_vllm(available["vllm"](), prompts, params_base)

        if not all_runs:
            print("nothing ran -- check --variants / installs / server")
            return {"benchmark": self.benchmark_name, "runs": []}

        self._check_agreement(all_runs, prompts)
        self._print_summary(all_runs, prompts)
        print(f"\nResults written to {self.config.output_dir}")
        return {"benchmark": self.benchmark_name, "runs": all_runs}

    def _run_tl(
        self,
        engine: PatchSweepEngine,
        tl_variants: list[str],
        prompts: list[str],
        params_base: dict[str, Any],
    ) -> list[dict[str, Any]]:
        print(f"\n[engine] tl variants={tl_variants}")
        engine.setup(self.config.model, dtype=self._opt("dtype", "bfloat16"))
        runs: list[dict[str, Any]] = []
        try:
            for pname in prompts:
                clean, corrupt = PROMPT_PAIRS[pname]
                for variant in tl_variants:
                    result = engine.run_sweep(
                        PatchSweepRequest(
                            clean=clean,
                            corrupt=corrupt,
                            answer=ANSWER,
                            variant=variant,
                            n_layers=int(self._opt("num_layers", 28)),
                            logits_chunk_budget=int(self._opt("logits_chunk_budget", 4096)),
                        )
                    )
                    row = {**result.to_dict(), "prompt": pname}
                    _print_run(f"{pname}/{result.variant}", result)
                    runs.append(row)
        finally:
            engine.teardown()
        self._write(engine, runs, params_base)
        return runs

    def _run_vllm(
        self,
        engine: PatchSweepEngine,
        prompts: list[str],
        params_base: dict[str, Any],
    ) -> list[dict[str, Any]]:
        print("\n[engine] vllm /v1/patch_sweep")
        try:
            engine.setup(self.config.model, base_url=self._opt("base_url"))
        except PatchSweepError as exc:
            print(f"  vllm_sweep: SKIPPED ({exc})")
            return []
        reps = int(self._opt("reps", 3))
        runs: list[dict[str, Any]] = []
        try:
            for pname in prompts:
                clean, corrupt = PROMPT_PAIRS[pname]
                for rep in range(reps):
                    result = engine.run_sweep(
                        PatchSweepRequest(
                            clean=clean,
                            corrupt=corrupt,
                            answer=ANSWER,
                            n_layers=int(self._opt("num_layers", 28)),
                        )
                    )
                    row = {**result.to_dict(), "prompt": pname, "rep": rep}
                    _print_run(f"{pname}/vllm_sweep rep{rep}", result)
                    runs.append(row)
        finally:
            engine.teardown()
        self._write(engine, runs, params_base)
        return runs

    def _write(
        self,
        engine: PatchSweepEngine,
        runs: list[dict[str, Any]],
        params_base: dict[str, Any],
    ) -> None:
        if not runs:
            return
        write_result(
            benchmark=f"{engine.name}.patch_sweep",
            parameters={**params_base, "engine": engine.name},
            results={"runs": runs},
            output_dir=self.config.output_dir,
            tag=self.config.tag,
            engine=engine.identity(),
        )

    @staticmethod
    def _check_agreement(runs: list[dict[str, Any]], prompts: list[str]) -> None:
        # Cross-tool agreement: every variant that ran a given prompt should find
        # the same causal site (position must match; the recovered plateau makes
        # layer ties within it expected).
        for pname in prompts:
            positions = {
                r["argmax"]["position"] for r in runs if r["prompt"] == pname
            }
            if len(positions) > 1:
                print(f"WARNING [{pname}]: argmax positions disagree: {positions}")

    @staticmethod
    def _print_summary(runs: list[dict[str, Any]], prompts: list[str]) -> None:
        print("\nSummary (cells/s):")
        for pname in prompts:
            rows = [r for r in runs if r["prompt"] == pname]
            if not rows:
                continue
            best: dict[str, float] = {}
            for r in rows:
                key = r["variant"]
                best[key] = max(best.get(key, 0.0), r["cells_per_s"])
            cells = rows[0]["cells"]
            parts = "  ".join(f"{k}={v:g}" for k, v in sorted(best.items()))
            print(f"  {pname:>6} ({cells} cells): {parts}")
