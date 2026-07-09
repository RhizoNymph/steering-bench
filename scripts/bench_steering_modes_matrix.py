#!/usr/bin/env python3
"""Cross-product benchmark of steering modes × workload dimensions.

Sweeps:

- ``mode`` ∈ {disabled, enabled_idle, inline_shared, inline_unique,
  named_shared, per_request_4}
- ``batch_size``
- ``num_hooks`` (1..3 hook points populated per request)
- ``num_layers_steered`` (subset of model layers; ``None`` = all)
- ``prompt_len``

Each cell runs ``bench_throughput.run_throughput`` so we capture both
batch wall-clock latency *and* tokens/sec from the same samples.
Results are written incrementally via the standard JSON schema in
``steering_bench.output.write_result`` so partial runs are useful and
``analyze.py`` can aggregate after the fact.

Three default subsets:

- ``--headline`` (default): small set that fits in ~30-40 minutes —
  enough to confirm the bench infrastructure works and the ranking
  across modes is what we expect.
- ``--mid``: ~3-4 hour sweep covering the headline modes across the
  most-interesting batch / prompt / hook combinations.
- ``--full``: ~12 hour sweep targeting the day-budget the user cares
  about.  Default modes × 5 batch sizes × 3 hook counts × 2 layer
  subsets × 3 prompt lengths = 540 cells.

All three are overridable on the CLI with ``--modes``, ``--batch-sizes``,
``--num-hooks-list``, ``--num-layers-steered-list``, ``--prompt-lens``.

Requires a vLLM build with steering support and (optimally) the
auto-promote helper from PR #145, otherwise ``inline_shared`` and
``per_request_4`` won't differ from ``inline_unique`` in practice.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch  # noqa: F401  imported for cuda OOM exception class

from steering_bench.engine.engines.vllm import VllmSteeringEngine
from steering_bench.harness.args import engine_names
from steering_bench.harness.models import get_model_config
from steering_bench.output import write_result

# Reuse the throughput primitive — gives both latency and tokens/sec
# per cell for free, sharing the LLM-init cost.  Resolved from the script
# directory, which Python places on sys.path[0] when this file is run.
from bench_throughput import run_throughput  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Subset presets
# ---------------------------------------------------------------------------

# Per-cell cost (gemma-3-4b-it / RTX 3090) ≈ 80 seconds wall:
# - LLM init: 30-50 s
# - warmup (3 iters) + measure (5 iters): ~30 s
# - teardown: a few s

PRESETS: dict[str, dict[str, list]] = {
    "headline": {
        "modes": [
            "disabled",
            "inline_shared",
            "inline_unique",
            "named_shared",
        ],
        "batch_sizes": [1, 8, 32],
        "num_hooks_list": [1],
        "num_layers_steered_list": [None],  # None = all layers
        "prompt_lens": [64, 256],
    },
    "mid": {
        "modes": [
            "disabled",
            "enabled_idle",
            "inline_shared",
            "inline_unique",
            "named_shared",
        ],
        "batch_sizes": [1, 4, 8, 16],
        "num_hooks_list": [1, 3],
        "num_layers_steered_list": [8, None],
        "prompt_lens": [64, 256],
    },
    "full": {
        "modes": [
            "disabled",
            "enabled_idle",
            "inline_shared",
            "inline_unique",
            "named_shared",
            "per_request_4",
        ],
        "batch_sizes": [1, 4, 8, 16, 32],
        "num_hooks_list": [1, 2, 3],
        "num_layers_steered_list": [8, None],
        "prompt_lens": [64, 256, 1024],
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_layer_subset_list(spec: str | None) -> list[int | None] | None:
    if spec is None:
        return None
    out: list[int | None] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in ("none", "all", "full"):
            out.append(None)
        else:
            out.append(int(token))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-product steering modes × workload bench"
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--output-dir", default="results/vllm/modes_matrix/")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="headline",
        help="Pick a default sweep size: headline (~30 min), mid (~3 hr), "
             "full (~12 hr).  Per-axis CLI flags override individual axes.",
    )
    parser.add_argument(
        "--modes", default=None,
        help="Override preset modes (comma-sep).",
    )
    parser.add_argument(
        "--batch-sizes", default=None,
        help="Override preset batch sizes (comma-sep ints).  Used as "
             "num_prompts to bench_throughput.run_throughput.",
    )
    parser.add_argument(
        "--num-hooks-list", default=None,
        help="Override preset num_hooks list (comma-sep ints in [1,3]).",
    )
    parser.add_argument(
        "--num-layers-steered-list", default=None,
        help="Override preset layer-subset list (comma-sep ints or "
             "'none'/'all' for full-model).",
    )
    parser.add_argument(
        "--prompt-lens", default=None,
        help="Override preset prompt lengths (comma-sep ints).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--max-steering-configs", type=int, default=None,
        help="Override the worker-side steering table capacity (passed "
             "to LLM(max_steering_configs=...)).",
    )
    parser.add_argument(
        "--disable-prefix-cache", action="store_true",
        help="Disable vLLM prefix caching (matches bench_latency / "
             "bench_throughput semantics).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the cell list and estimated wall time, then exit "
             "without running.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=engine_names(),
        help="Engine adapter. This cross-product exercises the vLLM-fork "
             "steering modes (named_shared, inline_unique, per_request_N); "
             "only --engine vllm is supported. For an engine-agnostic sweep "
             "use `python -m steering_bench run throughput --mode ...`.",
    )
    args = parser.parse_args()

    if args.engine != "vllm":
        parser.error(
            f"engine {args.engine!r} not supported: this script sweeps "
            "vLLM-fork steering modes with no engine-agnostic equivalent. Use "
            "--engine vllm, or `python -m steering_bench run throughput` for "
            "the portable subset."
        )

    preset = PRESETS[args.preset]
    modes = (
        [m.strip() for m in args.modes.split(",")] if args.modes
        else preset["modes"]
    )
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes
        else preset["batch_sizes"]
    )
    num_hooks_list = (
        [int(x) for x in args.num_hooks_list.split(",")]
        if args.num_hooks_list else preset["num_hooks_list"]
    )
    layer_subset_list = (
        _parse_layer_subset_list(args.num_layers_steered_list)
        if args.num_layers_steered_list else preset["num_layers_steered_list"]
    )
    prompt_lens = (
        [int(x) for x in args.prompt_lens.split(",")] if args.prompt_lens
        else preset["prompt_lens"]
    )

    config = get_model_config(args.model)
    engine_identity = VllmSteeringEngine().identity()

    # Build the cell list.  Skip combinations that would generate zero
    # additional information (e.g. layer-subset variation in a non-steering
    # mode is wasted work — collapse to a single rep).
    cells: list[dict] = []
    for mode in modes:
        is_steering = mode not in ("disabled", "enabled_idle")
        # Non-steering modes: layer/hook variation is degenerate; pick one rep.
        hooks_iter = num_hooks_list if is_steering else [num_hooks_list[0]]
        layers_iter = layer_subset_list if is_steering else [layer_subset_list[0]]
        for batch in batch_sizes:
            for prompt_len in prompt_lens:
                for num_hooks in hooks_iter:
                    for num_layers_steered in layers_iter:
                        cells.append({
                            "mode": mode,
                            "batch_size": batch,
                            "prompt_len": prompt_len,
                            "num_hooks": num_hooks,
                            "num_layers_steered": num_layers_steered,
                        })

    n_cells = len(cells)
    est_seconds = n_cells * 80  # rough per-cell wall on a 4B model / RTX 3090

    print("Steering-modes matrix bench")
    print(f"  model:               {args.model}")
    print(f"  preset:              {args.preset}")
    print(f"  modes:               {modes}")
    print(f"  batch_sizes:         {batch_sizes}")
    print(f"  num_hooks_list:      {num_hooks_list}")
    print(f"  num_layers_steered:  "
          f"{['all' if x is None else x for x in layer_subset_list]}")
    print(f"  prompt_lens:         {prompt_lens}")
    print(f"  cells:               {n_cells}")
    print(f"  est. wall:           {est_seconds // 60} min "
          f"({est_seconds // 3600}h {est_seconds % 3600 // 60}m)")
    print()

    if args.dry_run:
        print("--dry-run set; not executing.")
        return

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []
    started_at = time.time()

    for i, cell in enumerate(cells, 1):
        elapsed = time.time() - started_at
        eta = (elapsed / max(i - 1, 1)) * (n_cells - i + 1) if i > 1 else est_seconds
        print(f"\n[{i}/{n_cells}] mode={cell['mode']} batch={cell['batch_size']} "
              f"prompt={cell['prompt_len']} hooks={cell['num_hooks']} "
              f"layers={cell['num_layers_steered'] or 'all'}  "
              f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)")

        try:
            result = run_throughput(
                model=args.model,
                num_prompts=cell["batch_size"],
                prompt_len=cell["prompt_len"],
                max_tokens=args.max_tokens,
                mode=cell["mode"],
                warmup=args.warmup,
                iters=args.iters,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                max_steering_configs_override=args.max_steering_configs,
                enable_prefix_caching=not args.disable_prefix_cache,
                num_hooks=cell["num_hooks"],
                num_layers_steered=cell["num_layers_steered"],
            )
            err = None
            mean_tps = result["throughput_tokens_per_sec"]["mean_tps"]
            mean_ms = result["latency_ms"]["mean_ms"]
            print(f"    → {mean_tps:.0f} tok/s, batch_ms={mean_ms:.1f}")
        except torch.cuda.OutOfMemoryError:
            print("    → OOM (skipping)")
            result = {"error": "OOM"}
            err = "OOM"
        except Exception as exc:  # bench should not crash the whole sweep
            print(f"    → ERROR: {type(exc).__name__}: {exc}")
            result = {"error": f"{type(exc).__name__}: {exc}"}
            err = str(exc)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        params = {
            "model": args.model,
            "mode": cell["mode"],
            "batch_size": cell["batch_size"],
            "num_prompts": cell["batch_size"],
            "prompt_len": cell["prompt_len"],
            "max_tokens": args.max_tokens,
            "num_hooks": cell["num_hooks"],
            "num_layers_steered": (
                cell["num_layers_steered"]
                if cell["num_layers_steered"] is not None
                else config.num_layers
            ),
            "max_steering_configs_override": args.max_steering_configs,
            "prefix_caching": not args.disable_prefix_cache,
            "preset": args.preset,
        }
        results_out = {k: v for k, v in result.items() if k != "samples_ms"}
        write_result(
            benchmark="vllm.steering_modes_matrix",
            parameters=params,
            results=results_out,
            output_dir=args.output_dir,
            tag=args.tag,
            raw_samples_ms=result.get("samples_ms"),
            engine=engine_identity,
        )
        summary_rows.append({
            "cell": cell,
            "result": results_out,
            "error": err,
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    total_wall = time.time() - started_at
    print(f"\n{'=' * 110}")
    print(f"  Modes-Matrix Summary: {args.model}  (wall {total_wall/60:.1f} min)")
    print(f"{'=' * 110}")
    header = (
        f"{'mode':<18} {'batch':>5} {'prompt':>6} {'hooks':>5} {'layers':>6} "
        f"{'mean_ms':>10} {'p90_ms':>10} {'tok/s':>10} {'loss%':>7}"
    )
    print(header)
    print("-" * 110)
    for row in summary_rows:
        c = row["cell"]
        r = row["result"]
        if row["error"]:
            print(
                f"{c['mode']:<18} {c['batch_size']:>5} {c['prompt_len']:>6} "
                f"{c['num_hooks']:>5} "
                f"{c['num_layers_steered'] or 'all':>6} "
                f"{row['error']:>50}"
            )
            continue
        lat = r.get("latency_ms", {})
        tps = r.get("throughput_tokens_per_sec", {})
        loss = r.get("throughput_loss_pct")
        loss_str = f"{loss:.1f}" if loss is not None else "—"
        print(
            f"{c['mode']:<18} {c['batch_size']:>5} {c['prompt_len']:>6} "
            f"{c['num_hooks']:>5} "
            f"{(c['num_layers_steered'] or 'all'):>6} "
            f"{lat.get('mean_ms', 0):>10.1f} {lat.get('p90_ms', 0):>10.1f} "
            f"{tps.get('mean_tps', 0):>10.0f} {loss_str:>7}"
        )
    print("=" * 110)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
