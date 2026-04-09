# Analysis & Plotting

## Scope

Aggregate JSON results into a DataFrame and generate matplotlib charts.

### In Scope
- JSON result aggregation with schema validation
- Derived metrics (overhead_pct, speedup_vs_baseline, memory_overhead_mb)
- 4 chart types: overhead bars, library comparison, ablation heatmaps, scaling curves
- CSV export of full results table
- Text summary with headline numbers

### Not In Scope
- Interactive dashboards (plotly available but not used in MVP)
- Automated report generation (markdown/PDF)

## Data/Control Flow

```
scripts/analyze.py
  → aggregate(results_dir)
      → load_results(): walk dir, parse *.json, validate required keys
      → to_dataframe(): flatten parameters.* and results.* into columns
      → compute_derived(): add overhead_pct, speedup_vs_baseline columns
  → Plot functions (only run if data for that category exists):
      → plot_overhead_bars(): vLLM latency grouped bars
      → plot_library_comparison(): Tier 1 + Tier 2 horizontal bars
      → plot_ablation_heatmaps(): CUDA graphs 2x2 + hook points bars
      → plot_scaling_curves(): config count line plot with error bands
  → print_text_summary(): headline numbers to stdout
  → Export CSV
```

## Files

| File | Purpose |
|------|---------|
| `src/steering_bench/analysis/__init__.py` | Package marker |
| `src/steering_bench/analysis/aggregate.py` | JSON loading, DataFrame conversion, derived metrics |
| `scripts/analyze.py` | Plot generation + text summary |

## Invariants

- Only plots categories that have result data (graceful with partial runs)
- Flattens nested JSON with underscore separation (e.g. result_latency_ms_mean_ms)
- Skips raw sample arrays in DataFrame columns (too large)
- Uses non-interactive matplotlib backend (Agg)
- CSV export includes all records and derived columns
