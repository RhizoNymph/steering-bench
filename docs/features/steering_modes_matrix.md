# Steering Modes Matrix Bench

## Scope

In scope:

- Cross-product sweep of steering submission modes against four workload
  axes (`batch_size`, `num_hooks`, `num_layers_steered`, `prompt_len`).
- Single end-to-end script that produces the data needed to answer
  "does the steering implementation regress baseline / how much do the
  new optimizations close the inline-vs-named gap / what is the
  research-workload tax?"
- Reuses `bench_throughput.run_throughput` so each cell yields both
  batch-wall latency and tokens/sec from the same samples.

Out of scope:

- Microbenchmarks (kernel ops, hash + resolve costs) — those live in
  `bench_steering_op.py`, `bench_resolve_hash.py`, `bench_named_resolve.py`.
- Capture-consumer perf — see `docs/features/capture_consumers.md`.
- Cross-library comparison (TransformerLens, nnsight, etc.) — see
  `docs/features/external_comparison.md`.

## Modes

The catalog lives in
`scripts/bench_latency.py:run_mode` and is mirrored by
`scripts/bench_throughput.py:run_throughput`:

| mode | wire shape | what it measures |
|---|---|---|
| `disabled` | no steering subsystem | hard baseline |
| `enabled_idle` | steering on, no per-request vectors | fixed steering-on overhead |
| `inline_shared` | every request shares one `steering_vectors` (`[sp]*N`) | post-auto-promote steady state — auto-promote (vllm PR #145) lifts this to a named module on second sight |
| `inline_unique` | every request has a fresh seed | research-style workload — auto-promote can't dedup, every request ships its packed payload |
| `named_shared` | one module pre-registered via `register_steering_modules`, every request uses `steering_module_ref` | floor for the spec-reuse case — only `(name, scale)` rides the wire |
| `per_request_4` | cycle four distinct specs across the batch | partial reuse — exercise of the LRU dedup more directly |

The headline comparison is `inline_shared / named_shared` — the ratio
should be ≈ 1.0 once auto-promote engages.  `inline_unique` measures
the *unavoidable* per-request cost on workloads where every spec is
fresh.

## Running

```bash
# Quick smoke test — 24 cells, ~30 min on RTX 3090 / gemma-3-4b-it.
.venv/bin/python scripts/bench_steering_modes_matrix.py \
    --preset headline \
    --output-dir results/modes_matrix/

# Full sweep — 390 cells, ~9 hr on the same hardware.
.venv/bin/python scripts/bench_steering_modes_matrix.py \
    --preset full \
    --output-dir results/modes_matrix/

# Dry-run prints the cell list and estimated wall-time.
.venv/bin/python scripts/bench_steering_modes_matrix.py \
    --preset full --dry-run
```

Per-axis flags override the preset's individual axes
(`--modes`, `--batch-sizes`, `--num-hooks-list`,
`--num-layers-steered-list`, `--prompt-lens`).  See
`scripts/bench_steering_modes_matrix.py` for the full CLI.

## Presets

Preset cell counts at hooks ∈ `{1,2,3}` and `layers_steered ∈ {8, all}`:

| preset | modes | batch_sizes | hooks | layers | prompts | cells | est. wall (4B/3090) |
|---|---|---|---|---|---|---|---|
| `headline` | 4 | `[1, 8, 32]` | `[1]` | `[all]` | `[64, 256]` | 24 | ~30 min |
| `mid` | 5 | `[1, 4, 8, 16]` | `[1, 3]` | `[8, all]` | `[64, 256]` | 112 | ~2.5 hr |
| `full` | 6 | `[1, 4, 8, 16, 32]` | `[1, 2, 3]` | `[8, all]` | `[64, 256, 1024]` | 390 | ~9 hr |

The non-steering modes (`disabled`, `enabled_idle`) collapse the
`hooks × layers` axes to a single representative cell since variation
on those axes is degenerate.

## Output schema

Each cell writes a JSON record under `--output-dir` with
`benchmark = "vllm.steering_modes_matrix"`.  The result block carries
the standard `latency_ms.*` (mean, median, stddev, p10, p25, p50, p75,
p90, p99) and `throughput_tokens_per_sec.*` stats from
`steering_bench.timing.compute_stats`.  The parameter block carries
`mode`, `batch_size`, `num_prompts`, `prompt_len`, `max_tokens`,
`num_hooks`, `num_layers_steered` (the count, not the indices), and
provenance metadata (`preset`, `tag`, model dims).

## Analysis

`scripts/analyze.py` picks up the matrix records automatically:

- `plot_steering_modes_matrix` renders a 2×N grid (rows = latency / tok-s,
  cols = unique `(num_hooks, num_layers_steered, prompt_len)` panels)
  with one line per mode.
- `print_steering_modes_matrix_summary` prints a per-cell text table
  plus the `inline_shared / named_shared` ratio per batch size.

Error rows are filtered out so re-runs of failed cells overwrite
cleanly.

## Invariants

- `inline_shared` ≤ `named_shared × 1.02` after the auto-promote helper
  (vllm PR #145) lands.  A drift above that range is a regression.
- `inline_unique` ≥ `inline_shared` by some non-zero amount at
  `batch_size ≥ 2` — because each unique spec defeats the LRU dedup.
  At `batch_size = 1` the two collapse (one request can't dedup against
  itself).
- `disabled` ≤ every other mode.  A regression below disabled means
  the steering subsystem somehow improved generation; investigate before
  trusting the numbers.

## Files

- `scripts/bench_steering_modes_matrix.py` — entry point.  Defines
  `PRESETS` and dispatches to `bench_throughput.run_throughput` per
  cell.
- `scripts/bench_throughput.py` — provides `run_throughput(mode=...)`.
  Mode catalog moved here in the modes-matrix change; legacy
  `--configs-sweep` integers are translated via
  `_legacy_mode_for_distinct`.
- `scripts/bench_latency.py` — same mode catalog as the throughput
  script, useful when you want only per-call wall-clock without the
  tokens/sec math.
- `src/steering_bench/vectors.py` — `random_steering_vectors` accepts
  a `layer_subset` argument; `even_layer_subset(num_layers, count)`
  picks evenly-spaced indices for the layer-count sweep.
- `scripts/analyze.py` — `plot_steering_modes_matrix` and
  `print_steering_modes_matrix_summary` produce the comparison views.
