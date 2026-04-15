#!/usr/bin/env python3
"""Aggregate benchmark results and generate plots.

Reads results/**/*.json, computes derived metrics, and produces
matplotlib charts suitable for an article. Skips plot categories
that have no data so this can be run incrementally as benchmarks
complete.

Covers all benchmark types in the suite:
    vllm.latency, vllm.throughput, vllm.throughput_matrix,
    vllm.memory, vllm.mixed_batch
    ablation.cuda_graphs, ablation.hook_points, ablation.config_scaling
    micro.steering_op, micro.steering_manager, micro.index_building
    external.tier1.*, external.tier2.*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steering_bench.analysis.aggregate import aggregate


# ── Styling ───────────────────────────────────────────────────────────────────

COLORS = {
    "disabled": "#2196F3",
    "enabled_idle": "#4CAF50",
    "mixed_25": "#FFC107",
    "mixed_50": "#FF9800",
    "mixed_75": "#FF5722",
    "per_request_1": "#E91E63",
    "per_request_4": "#9C27B0",
    "all_steered": "#B71C1C",
    "hf_baseline": "#9E9E9E",
    "transformerlens": "#E91E63",
    "nnsight": "#9C27B0",
    "repeng": "#FF5722",
    "pyvene": "#795548",
    "vllm_single": "#2196F3",
    "vllm_batched": "#00BCD4",
    "1_hook": "#4CAF50",
    "2_hooks": "#FF9800",
    "3_hooks": "#F44336",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig, path: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")


def _tps_field(row: pd.Series, stat: str = "mean") -> float | None:
    """Extract throughput stat, handling old mean_ms and new mean_tps schemas."""
    new_key = f"result_throughput_tokens_per_sec_{stat}_tps"
    old_key = f"result_throughput_tokens_per_sec_{stat}_ms"
    if new_key in row and pd.notna(row[new_key]):
        return row[new_key]
    if old_key in row and pd.notna(row[old_key]):
        return row[old_key]
    return None


# Model shape lookup for theoretical memory recomputation. Early memory
# benchmark runs wrote theoretical_mb using an fp32 formula; we recompute
# freshly here for bf16 instead of trusting the stored value.
MODEL_SHAPES = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "meta-llama/Llama-3.1-8B": {"hidden_size": 4096, "num_layers": 32},
    "meta-llama/Llama-3.1-70B": {"hidden_size": 8192, "num_layers": 80},
}


def theoretical_memory_mb(
    model: str, max_configs: int, dtype_bytes: int = 2
) -> float | None:
    """Recompute theoretical steering buffer memory in MB (bf16 by default).

    Formula: 3 hooks × (max_configs + 3) × hidden_size × dtype_bytes × num_layers.
    Returns None if model shape is unknown or max_configs is 0/invalid.
    """
    shape = MODEL_SHAPES.get(model)
    if shape is None or not max_configs or max_configs <= 0:
        return None
    bytes_total = (
        3 * (max_configs + 3) * shape["hidden_size"] * dtype_bytes * shape["num_layers"]
    )
    return bytes_total / (1024 * 1024)


# ── vLLM latency ──────────────────────────────────────────────────────────────


def plot_overhead_bars(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """vLLM latency: grouped bars by steering mode across batch sizes."""
    data = df[df["benchmark"] == "vllm.latency"].copy()
    if data.empty:
        return

    # Prefer prefix_cache=off runs if available (cleaner baseline)
    if "param_prefix_caching" in data.columns:
        pref_off = data[data["param_prefix_caching"] == False]  # noqa: E712
        if not pref_off.empty:
            data = pref_off
            suffix = "_no_prefix_cache"
        else:
            suffix = ""
    else:
        suffix = ""

    modes = ["disabled", "enabled_idle", "per_request_1", "per_request_4"]
    batch_sizes = sorted(data["param_batch_size"].dropna().unique())
    if not batch_sizes:
        return

    fig, ax = plt.subplots()
    x = np.arange(len(batch_sizes))
    width = 0.18
    offsets = np.arange(len(modes)) - (len(modes) - 1) / 2

    for i, mode in enumerate(modes):
        mode_data = data[data["param_mode"] == mode]
        means = []
        stds = []
        for bs in batch_sizes:
            row = mode_data[mode_data["param_batch_size"] == bs]
            if not row.empty:
                means.append(row.iloc[0].get("result_latency_ms_mean_ms", 0))
                stds.append(row.iloc[0].get("result_latency_ms_stddev_ms", 0))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(
            x + offsets[i] * width,
            means,
            width,
            yerr=stds,
            label=mode,
            color=COLORS.get(mode, "#666"),
            capsize=3,
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"vLLM Latency: Steering Overhead by Mode{' (prefix cache off)' if suffix else ''}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    ax.legend()
    _save(fig, output_dir / f"overhead_bars{suffix}.{fmt}", f"overhead_bars{suffix}.{fmt}")


# ── vLLM throughput (config sweep) ────────────────────────────────────────────


def plot_throughput_by_configs(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Throughput vs distinct_configs, one line per tag (e.g. baseline, big-table)."""
    data = df[df["benchmark"] == "vllm.throughput"].copy()
    if data.empty:
        return

    # Group by tag to overlay different experiments
    tags = sorted(data["tag"].dropna().unique())
    tags = [t for t in tags if t]  # drop empty tags
    if not tags:
        tags = [""]  # fall back to untagged data

    fig, ax = plt.subplots()
    for tag in tags:
        tag_data = data[data["tag"] == tag].sort_values("param_distinct_configs")
        if tag_data.empty:
            continue
        x = tag_data["param_distinct_configs"].values
        y = [_tps_field(r) for _, r in tag_data.iterrows()]
        label = tag if tag else "untagged"
        ax.plot(x, y, "o-", label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Distinct steering configs per batch")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs distinct configs")
    ax.legend()
    _save(fig, output_dir / f"throughput_by_configs.{fmt}", f"throughput_by_configs.{fmt}")


# ── Throughput matrix (mode × batch) — 4 plots covering both latency & throughput


def _matrix_cell_value(data: pd.DataFrame, mode: str, bs, metric: str) -> float | None:
    """Get a single value from the throughput matrix dataset.

    metric: one of "latency" (mean_ms) or "throughput" (mean_tps)
    """
    row = data[(data["param_mode"] == mode) & (data["param_batch_size"] == bs)]
    if row.empty:
        return None
    r = row.iloc[0]
    if metric == "latency":
        return r.get("result_latency_ms_mean_ms")
    if metric == "throughput":
        return _tps_field(r)
    return None


def _grouped_bars(
    data: pd.DataFrame,
    modes_present: list[str],
    batch_sizes: list,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(batch_sizes))
    width = 0.8 / max(1, len(modes_present))
    offsets = np.arange(len(modes_present)) - (len(modes_present) - 1) / 2

    for i, mode in enumerate(modes_present):
        vals = [_matrix_cell_value(data, mode, bs, metric) or 0 for bs in batch_sizes]
        ax.bar(
            x + offsets[i] * width,
            vals,
            width,
            label=mode,
            color=COLORS.get(mode, "#666"),
        )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    ax.legend(loc="upper left")
    _save(fig, output_path, name)


def _loss_heatmap(
    data: pd.DataFrame,
    modes_present: list[str],
    batch_sizes: list,
    metric: str,
    label: str,
    title: str,
    output_path: Path,
    name: str,
) -> None:
    """Heatmap of percentage delta vs disabled (positive = cost)."""
    matrix = np.full((len(modes_present), len(batch_sizes)), np.nan)
    for i, mode in enumerate(modes_present):
        for j, bs in enumerate(batch_sizes):
            mode_val = _matrix_cell_value(data, mode, bs, metric)
            base_val = _matrix_cell_value(data, "disabled", bs, metric)
            if mode_val is None or base_val is None or base_val <= 0:
                continue
            if metric == "latency":
                # Overhead % (higher = worse)
                matrix[i, j] = (mode_val - base_val) / base_val * 100
            else:
                # Throughput loss % (higher = worse)
                matrix[i, j] = (base_val - mode_val) / base_val * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = max(100, np.nanmax(matrix)) if not np.isnan(matrix).all() else 100
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    ax.set_yticks(range(len(modes_present)))
    ax.set_yticklabels(modes_present)
    ax.set_xlabel("Batch Size")
    ax.set_title(title)
    for i in range(len(modes_present)):
        for j in range(len(batch_sizes)):
            if not np.isnan(matrix[i, j]):
                threshold = vmax * 0.5
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:+.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black" if matrix[i, j] < threshold else "white",
                )
    fig.colorbar(im, label=label)
    _save(fig, output_path, name)


def plot_throughput_matrix(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Throughput matrix benchmark: 4 plots covering latency and throughput.

    Emits one set of 4 files per distinct ``tag`` value found in the data.
    With a single tag (or no tag), file names are ``matrix_*.{fmt}``.
    With multiple tags, file names are ``matrix_*_<tag>.{fmt}`` so each
    set can be placed independently in the article.
    """
    data = df[df["benchmark"] == "vllm.throughput_matrix"].copy()
    if data.empty:
        return

    if "tag" not in data.columns:
        data["tag"] = ""

    mode_order = [
        "disabled",
        "enabled_idle",
        "mixed_25",
        "mixed_50",
        "mixed_75",
        "all_steered",
    ]

    all_tags = sorted(data["tag"].fillna("").unique(), key=lambda t: (t != "", t))
    tag_groups = [
        (tag, data[data["tag"].fillna("") == tag])
        for tag in all_tags
    ]
    tag_groups = [(t, g) for t, g in tag_groups if not g.empty]

    if not tag_groups:
        return

    multiple_tags = len(tag_groups) > 1

    for tag, tag_data in tag_groups:
        batch_sizes = sorted(tag_data["param_batch_size"].dropna().unique())
        if not batch_sizes:
            continue

        modes_present = [
            m for m in mode_order if not tag_data[tag_data["param_mode"] == m].empty
        ]

        if multiple_tags:
            safe_tag = (tag or "untagged").replace("/", "_").replace(" ", "_")
            suffix = f"_{safe_tag}"
        else:
            suffix = ""

        # 1. Throughput grouped bars
        _grouped_bars(
            tag_data,
            modes_present,
            batch_sizes,
            metric="throughput",
            ylabel="Throughput (tokens/sec)",
            title="Throughput matrix: mode × batch_size",
            output_path=output_dir / f"matrix_throughput_bars{suffix}.{fmt}",
            name=f"matrix_throughput_bars{suffix}.{fmt}",
        )

        # 2. Throughput loss heatmap
        _loss_heatmap(
            tag_data,
            modes_present,
            batch_sizes,
            metric="throughput",
            label="throughput loss (%)",
            title="Throughput loss (%) vs disabled",
            output_path=output_dir / f"matrix_throughput_heatmap{suffix}.{fmt}",
            name=f"matrix_throughput_heatmap{suffix}.{fmt}",
        )

        # 3. Latency grouped bars
        _grouped_bars(
            tag_data,
            modes_present,
            batch_sizes,
            metric="latency",
            ylabel="Latency (ms)",
            title="Latency matrix: mode × batch_size",
            output_path=output_dir / f"matrix_latency_bars{suffix}.{fmt}",
            name=f"matrix_latency_bars{suffix}.{fmt}",
        )

        # 4. Latency overhead heatmap
        _loss_heatmap(
            tag_data,
            modes_present,
            batch_sizes,
            metric="latency",
            label="latency overhead (%)",
            title="Latency overhead (%) vs disabled",
            output_path=output_dir / f"matrix_latency_heatmap{suffix}.{fmt}",
            name=f"matrix_latency_heatmap{suffix}.{fmt}",
        )


# ── Mixed batch (proportional scaling) ────────────────────────────────────────


def plot_max_tokens_sweep(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Per-step latency vs max_tokens, one line per num_active.

    Demonstrates that the per-step overhead converges to the populate floor
    as max_tokens grows. The submission-cost component scales as 1/max_tokens
    while the populate component is constant per step, so each curve looks
    like ``floor + per_active_submission * num_active / max_tokens``.

    The n=0 (idle) line is the floor; higher num_active lines start above
    the floor at small max_tokens and converge to it as max_tokens grows.

    Emits one file per distinct ``tag`` value found in the data. With a
    single tag (or no tag), output is ``max_tokens_sweep.{fmt}``. With
    multiple tags, output is ``max_tokens_sweep_<tag>.{fmt}``.
    """
    data = df[df["benchmark"] == "vllm.max_tokens_sweep"].copy()
    if data.empty:
        return

    if "param_max_tokens" not in data.columns or "param_num_active" not in data.columns:
        return

    data = data.dropna(subset=["param_max_tokens", "param_num_active"])
    if data.empty:
        return

    if "tag" not in data.columns:
        data["tag"] = ""

    # Compute per-step latency from the raw latency column. Some runs may
    # already have a ``result_per_step_ms`` field; fall back to it if so.
    if "result_per_step_ms" in data.columns and data["result_per_step_ms"].notna().any():
        data["per_step_ms"] = data["result_per_step_ms"]
    else:
        data["per_step_ms"] = data["result_latency_ms_mean_ms"] / data["param_max_tokens"]

    all_tags = sorted(data["tag"].fillna("").unique(), key=lambda t: (t != "", t))
    tag_groups = [
        (tag, data[data["tag"].fillna("") == tag])
        for tag in all_tags
    ]
    tag_groups = [(t, g) for t, g in tag_groups if not g.empty]

    if not tag_groups:
        return

    multiple_tags = len(tag_groups) > 1

    for tag, tag_data in tag_groups:
        num_active_values = sorted(tag_data["param_num_active"].dropna().unique())
        max_tokens_values = sorted(tag_data["param_max_tokens"].dropna().unique())
        if not num_active_values or not max_tokens_values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left panel: per-step latency vs max_tokens (the main "convergence" plot)
        cmap = plt.get_cmap("viridis")
        for i, n in enumerate(num_active_values):
            sub = (
                tag_data[tag_data["param_num_active"] == n]
                .sort_values("param_max_tokens")
            )
            if sub.empty:
                continue
            color = cmap(i / max(1, len(num_active_values) - 1))
            label = f"num_active={int(n)}"
            axes[0].plot(
                sub["param_max_tokens"].values,
                sub["per_step_ms"].values,
                "o-",
                color=color,
                label=label,
                linewidth=2,
                markersize=7,
            )

        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("max_tokens")
        axes[0].set_ylabel("Per-step latency (ms)")
        axes[0].set_title(
            "Per-step latency vs max_tokens\n"
            "(submission cost amortizes as outputs grow)"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(title="active steered", loc="best")

        # Right panel: overhead vs idle (n=0) baseline as a percentage
        if 0 in num_active_values:
            idle = (
                tag_data[tag_data["param_num_active"] == 0]
                .set_index("param_max_tokens")["per_step_ms"]
            )
            for i, n in enumerate(num_active_values):
                if n == 0:
                    continue
                sub = (
                    tag_data[tag_data["param_num_active"] == n]
                    .sort_values("param_max_tokens")
                    .copy()
                )
                sub["overhead_pct"] = (
                    100.0
                    * (
                        sub["per_step_ms"].values
                        - idle.reindex(sub["param_max_tokens"]).values
                    )
                    / idle.reindex(sub["param_max_tokens"]).values
                )
                color = cmap(i / max(1, len(num_active_values) - 1))
                axes[1].plot(
                    sub["param_max_tokens"].values,
                    sub["overhead_pct"].values,
                    "o-",
                    color=color,
                    label=f"num_active={int(n)}",
                    linewidth=2,
                    markersize=7,
                )
            axes[1].set_xscale("log", base=2)
            axes[1].set_xlabel("max_tokens")
            axes[1].set_ylabel("Per-step overhead vs idle (%)")
            axes[1].set_title(
                "Steering overhead %\n"
                "(decays toward 0 as max_tokens grows)"
            )
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.5)
            axes[1].legend(title="active steered", loc="best")
        else:
            axes[1].text(
                0.5,
                0.5,
                "n=0 baseline missing\n(re-run with --num-active-list including 0)",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )

        fig.tight_layout()

        if multiple_tags:
            safe_tag = (tag or "untagged").replace("/", "_").replace(" ", "_")
            filename = f"max_tokens_sweep_{safe_tag}.{fmt}"
        else:
            filename = f"max_tokens_sweep.{fmt}"

        _save(fig, output_dir / filename, filename)


def plot_mixed_batch(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Mixed batch: latency and throughput vs num_active within a fixed batch size.

    Emits one file per distinct ``tag`` value found in the data. With a
    single tag (or no tag), output is ``mixed_batch.{fmt}``. With multiple
    tags, output is ``mixed_batch_<tag>.{fmt}``, one per tag, so the article
    can place them wherever it likes without being locked into a multi-panel
    layout.
    """
    data = df[df["benchmark"] == "vllm.mixed_batch"].copy()
    if data.empty:
        return

    if "tag" not in data.columns:
        data["tag"] = ""

    # Distinct tags, in a stable order. Empty tag sorts first; otherwise
    # alphabetical. Drop any tag with zero records.
    all_tags = sorted(data["tag"].fillna("").unique(), key=lambda t: (t != "", t))
    tag_groups = [
        (tag, data[data["tag"].fillna("") == tag])
        for tag in all_tags
    ]
    tag_groups = [(t, g) for t, g in tag_groups if not g.empty]

    if not tag_groups:
        return

    multiple_tags = len(tag_groups) > 1

    for tag, tag_data in tag_groups:
        batch_sizes = sorted(tag_data["param_batch_size"].dropna().unique())
        if not batch_sizes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for bs in batch_sizes:
            bs_data = (
                tag_data[tag_data["param_batch_size"] == bs]
                .sort_values("param_num_active")
            )
            x = bs_data["param_num_active"].values
            lat = bs_data["result_latency_ms_mean_ms"].values
            tps = [_tps_field(r) for _, r in bs_data.iterrows()]
            label = f"batch={int(bs)}"
            axes[0].plot(x, lat, "o-", label=label, linewidth=2, markersize=6)
            axes[1].plot(x, tps, "o-", label=label, linewidth=2, markersize=6)

        axes[0].set_xlabel("Number of steered requests in batch")
        axes[0].set_ylabel("Batch latency (ms)")
        axes[0].set_title("Mixed-batch latency: proportional to active count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Number of steered requests in batch")
        axes[1].set_ylabel("Throughput (tokens/sec)")
        axes[1].set_title("Mixed-batch throughput: proportional to active count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()

        # File naming: include tag suffix only when there are multiple tags
        # so single-tag runs produce the conventional ``mixed_batch.png``.
        if multiple_tags:
            safe_tag = tag if tag else "untagged"
            # Replace path-unfriendly characters
            safe_tag = safe_tag.replace("/", "_").replace(" ", "_")
            filename = f"mixed_batch_{safe_tag}.{fmt}"
        else:
            filename = f"mixed_batch.{fmt}"

        _save(fig, output_dir / filename, filename)


# ── Table sizing (max_cfg × batch × distinct) ─────────────────────────────────


def plot_table_sizing(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Table sizing: throughput loss vs distinct, with max_cfg as line per panel.

    Splits by tag so a refresh under a new tag (e.g. ``postfix-populate``)
    doesn't get layered on top of the old run in the same chart.
    """
    all_data = df[df["benchmark"] == "vllm.table_sizing"].copy()
    if all_data.empty:
        return

    all_tags = sorted(all_data["tag"].fillna("").unique(), key=lambda t: (t != "", t))
    tag_groups = [
        (tag, all_data[all_data["tag"].fillna("") == tag]) for tag in all_tags
    ]
    tag_groups = [(t, g) for t, g in tag_groups if not g.empty]
    if not tag_groups:
        return
    multiple_tags = len(tag_groups) > 1

    for tag, data in tag_groups:
        if multiple_tags:
            safe_tag = (tag or "untagged").replace("/", "_").replace(" ", "_")
            suffix = f"_{safe_tag}"
        else:
            suffix = ""
        _plot_table_sizing_one(data, output_dir, fmt, suffix)


def _plot_table_sizing_one(
    data: pd.DataFrame, output_dir: Path, fmt: str, suffix: str
) -> None:
    # Baselines: mode=disabled rows (max_steering_configs=0)
    baseline_rows = data[data["param_mode"] == "disabled"]
    disabled_tps: dict[float, float] = {}
    disabled_lat: dict[float, float] = {}
    for _, r in baseline_rows.iterrows():
        bs = r["param_batch_size"]
        tps = _tps_field(r)
        lat = r.get("result_latency_ms_mean_ms")
        if tps and pd.notna(bs):
            disabled_tps[bs] = tps
        if lat and pd.notna(bs):
            disabled_lat[bs] = lat

    steered = data[data["param_mode"] == "steered"].copy()
    if steered.empty:
        return

    batch_sizes = sorted(steered["param_batch_size"].dropna().unique())
    max_cfgs = sorted(steered["param_max_steering_configs"].dropna().unique())
    if not batch_sizes or not max_cfgs:
        return

    # Panel colors per max_cfg
    max_cfg_colors = {
        int(mc): plt.cm.viridis(i / max(1, len(max_cfgs) - 1))
        for i, mc in enumerate(max_cfgs)
    }

    # ── Throughput loss % panels ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 5), sharey=True)
    if len(batch_sizes) == 1:
        axes = [axes]

    for ax, bs in zip(axes, batch_sizes):
        baseline = disabled_tps.get(bs)
        for mc in max_cfgs:
            rows = steered[
                (steered["param_batch_size"] == bs)
                & (steered["param_max_steering_configs"] == mc)
            ].sort_values("param_distinct_configs")
            if rows.empty:
                continue
            x = rows["param_distinct_configs"].values
            tps_vals = [_tps_field(r) for _, r in rows.iterrows()]
            if baseline and baseline > 0:
                y = [
                    ((baseline - t) / baseline * 100) if t else None
                    for t in tps_vals
                ]
            else:
                y = tps_vals
            ax.plot(
                x,
                y,
                "o-",
                label=f"max_cfg={int(mc)}",
                color=max_cfg_colors[int(mc)],
                linewidth=2,
                markersize=7,
            )

        ax.set_title(f"batch={int(bs)}")
        ax.set_xlabel("distinct steering configs in batch")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel(
                "Throughput loss vs disabled (%)"
                if disabled_tps
                else "Throughput (tokens/sec)"
            )
        ax.legend()

    fig.suptitle(
        "Table sizing: throughput loss vs distinct configs, per batch size",
        fontsize=13,
    )
    _save(
        fig,
        output_dir / f"table_sizing_throughput{suffix}.{fmt}",
        f"table_sizing_throughput{suffix}.{fmt}",
    )

    # ── Latency overhead % panels ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5 * len(batch_sizes), 5), sharey=True)
    if len(batch_sizes) == 1:
        axes = [axes]

    for ax, bs in zip(axes, batch_sizes):
        baseline_ms = disabled_lat.get(bs)
        for mc in max_cfgs:
            rows = steered[
                (steered["param_batch_size"] == bs)
                & (steered["param_max_steering_configs"] == mc)
            ].sort_values("param_distinct_configs")
            if rows.empty:
                continue
            x = rows["param_distinct_configs"].values
            lat_vals = rows["result_latency_ms_mean_ms"].values
            if baseline_ms and baseline_ms > 0:
                y = [(lat - baseline_ms) / baseline_ms * 100 for lat in lat_vals]
            else:
                y = lat_vals
            ax.plot(
                x,
                y,
                "o-",
                label=f"max_cfg={int(mc)}",
                color=max_cfg_colors[int(mc)],
                linewidth=2,
                markersize=7,
            )

        ax.set_title(f"batch={int(bs)}")
        ax.set_xlabel("distinct steering configs in batch")
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel(
                "Latency overhead vs disabled (%)"
                if disabled_lat
                else "Latency (ms)"
            )
        ax.legend()

    fig.suptitle(
        "Table sizing: latency overhead vs distinct configs, per batch size",
        fontsize=13,
    )
    _save(
        fig,
        output_dir / f"table_sizing_latency{suffix}.{fmt}",
        f"table_sizing_latency{suffix}.{fmt}",
    )


# ── Memory ────────────────────────────────────────────────────────────────────


def plot_memory_scaling(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Memory cost vs max_steering_configs, measured vs theoretical (bf16)."""
    data = df[df["benchmark"] == "vllm.memory"].copy()
    if data.empty:
        return

    data = data.sort_values("param_max_steering_configs")
    configs = data["param_max_steering_configs"].values
    delta = data.get("result_steering_delta_mb")
    if delta is None:
        return

    # Recompute theoretical in bf16 per row (fresh, not from stored value)
    theoretical_vals = []
    for _, row in data.iterrows():
        model = row.get("param_model", "google/gemma-3-4b-it")
        mc = row.get("param_max_steering_configs")
        t = theoretical_memory_mb(model, int(mc) if pd.notna(mc) else 0)
        theoretical_vals.append(t if t is not None else np.nan)

    fig, ax = plt.subplots()
    ax.plot(
        configs, delta.values, "o-", label="Measured", linewidth=2, markersize=7, color="#2196F3"
    )
    if any(v is not None and not np.isnan(v) for v in theoretical_vals):
        ax.plot(
            configs,
            theoretical_vals,
            "s--",
            label="Theoretical (bf16)",
            linewidth=2,
            markersize=7,
            color="#FF9800",
        )
    ax.set_xlabel("max_steering_configs")
    ax.set_ylabel("Steering buffer memory (MB)")
    ax.set_title("Memory scaling: measured vs theoretical (Gemma-3-4B, bf16)")
    ax.legend()
    _save(fig, output_dir / f"memory_scaling.{fmt}", f"memory_scaling.{fmt}")


# ── CUDA graphs ablation ──────────────────────────────────────────────────────


def plot_cuda_graphs_ablation(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """CUDA graphs ablation: eager vs graphs across batch sizes.

    Produces:
      - cuda_graphs_latency.png: 4 lines (eager/graphs × steer on/off), latency vs batch
      - cuda_graphs_throughput.png: same 4 lines, throughput vs batch
      - cuda_graphs_speedup.png: graph speedup ratio, with vs without steering
      - per_step_overhead.png: per-step steering overhead vs batch size
    """
    data = df[df["benchmark"] == "ablation.cuda_graphs"].copy()
    if data.empty:
        return

    batch_sizes = sorted(data["param_batch_size"].dropna().unique())
    if not batch_sizes:
        return

    # Extract (latency, throughput) per (batch_size, enforce_eager, enable_steering)
    def get_row(bs, eager: bool, steer: bool):
        row = data[
            (data["param_batch_size"] == bs)
            & (data["param_enforce_eager"] == eager)
            & (data["param_enable_steering"] == steer)
        ]
        if row.empty:
            return None
        r = row.iloc[0]
        lat = r.get("result_latency_ms_mean_ms")
        if lat is None or pd.isna(lat):
            return None
        # Derive throughput from latency + parameters
        prompt_len = r.get("param_prompt_len", 64)
        max_tokens = r.get("param_max_tokens", 128)
        total_tokens = bs * (prompt_len + max_tokens)
        tps = total_tokens / (lat / 1000.0) if lat > 0 else None
        return {"latency": lat, "throughput": tps}

    # Build per-configuration series
    configs = [
        ("graphs_no_steer", False, False, "#2196F3", "-"),
        ("graphs_w_steer", False, True, "#F44336", "-"),
        ("eager_no_steer", True, False, "#2196F3", "--"),
        ("eager_w_steer", True, True, "#F44336", "--"),
    ]

    series: dict[str, dict] = {}
    for label, eager, steer, color, style in configs:
        lat_vals = []
        tps_vals = []
        for bs in batch_sizes:
            cell = get_row(bs, eager, steer)
            lat_vals.append(cell["latency"] if cell else np.nan)
            tps_vals.append(cell["throughput"] if cell else np.nan)
        series[label] = {
            "latency": lat_vals,
            "throughput": tps_vals,
            "color": color,
            "style": style,
        }

    # ── 4-line latency chart ────────────────────────────────────────────────
    fig, ax = plt.subplots()
    for label, s in series.items():
        ax.plot(
            batch_sizes,
            s["latency"],
            marker="o",
            linestyle=s["style"],
            color=s["color"],
            label=label.replace("_", " "),
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CUDA graphs × steering: latency vs batch size")
    ax.set_xscale("log", base=2)
    ax.legend()
    _save(fig, output_dir / f"cuda_graphs_latency.{fmt}", f"cuda_graphs_latency.{fmt}")

    # ── 4-line throughput chart ─────────────────────────────────────────────
    fig, ax = plt.subplots()
    for label, s in series.items():
        ax.plot(
            batch_sizes,
            s["throughput"],
            marker="o",
            linestyle=s["style"],
            color=s["color"],
            label=label.replace("_", " "),
            linewidth=2,
            markersize=7,
        )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("CUDA graphs × steering: throughput vs batch size")
    ax.set_xscale("log", base=2)
    ax.legend()
    _save(fig, output_dir / f"cuda_graphs_throughput.{fmt}", f"cuda_graphs_throughput.{fmt}")

    # ── Graph speedup ratio (headline interaction chart) ────────────────────
    # speedup_no_steer = eager_no_steer / graphs_no_steer
    # speedup_w_steer  = eager_w_steer  / graphs_w_steer
    speedup_no_steer = []
    speedup_w_steer = []
    for i, bs in enumerate(batch_sizes):
        g_ns = series["graphs_no_steer"]["latency"][i]
        g_ws = series["graphs_w_steer"]["latency"][i]
        e_ns = series["eager_no_steer"]["latency"][i]
        e_ws = series["eager_w_steer"]["latency"][i]
        speedup_no_steer.append(e_ns / g_ns if g_ns and not np.isnan(g_ns) else np.nan)
        speedup_w_steer.append(e_ws / g_ws if g_ws and not np.isnan(g_ws) else np.nan)

    fig, ax = plt.subplots()
    ax.plot(
        batch_sizes,
        speedup_no_steer,
        "o-",
        color="#2196F3",
        label="without steering",
        linewidth=2,
        markersize=8,
    )
    ax.plot(
        batch_sizes,
        speedup_w_steer,
        "s-",
        color="#F44336",
        label="with steering",
        linewidth=2,
        markersize=8,
    )
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="no speedup")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("CUDA graph speedup (eager / graphs)")
    ax.set_title("CUDA graph benefit: preserved without steering, degraded with steering")
    ax.set_xscale("log", base=2)
    ax.legend()
    _save(fig, output_dir / f"cuda_graphs_speedup.{fmt}", f"cuda_graphs_speedup.{fmt}")

    # ── Per-step steering overhead (keep this one, it's the headline chart) ─
    graphs_deltas = []
    eager_deltas = []
    for i, bs in enumerate(batch_sizes):
        g_ns = series["graphs_no_steer"]["latency"][i]
        g_ws = series["graphs_w_steer"]["latency"][i]
        e_ns = series["eager_no_steer"]["latency"][i]
        e_ws = series["eager_w_steer"]["latency"][i]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [g_ns, g_ws, e_ns, e_ws]):
            continue
        steps = 128  # approximate decode steps (max_tokens=128 default)
        graphs_deltas.append((bs, (g_ws - g_ns) / steps))
        eager_deltas.append((bs, (e_ws - e_ns) / steps))

    if graphs_deltas and eager_deltas:
        fig, ax = plt.subplots()
        gbs, gvals = zip(*graphs_deltas)
        ebs, evals = zip(*eager_deltas)
        ax.plot(gbs, gvals, "o-", label="with CUDA graphs", linewidth=2, markersize=8, color="#2196F3")
        ax.plot(ebs, evals, "s--", label="eager (no graphs)", linewidth=2, markersize=8, color="#F44336")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Per-step steering overhead (ms)")
        ax.set_title("Per-step steering overhead scales with batch size")
        ax.set_xscale("log", base=2)
        ax.legend()
        _save(fig, output_dir / f"per_step_overhead.{fmt}", f"per_step_overhead.{fmt}")


# ── Hook points ablation ──────────────────────────────────────────────────────


def plot_hook_points(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Hook point count vs latency (linear scaling finding)."""
    data = df[df["benchmark"] == "ablation.hook_points"].copy()
    if data.empty:
        return

    batch_sizes = sorted(data["param_batch_size"].dropna().unique())
    hook_configs = sorted(data["param_hook_config"].dropna().unique())
    if not batch_sizes or not hook_configs:
        return

    # Line plot showing linear scaling
    fig, ax = plt.subplots()
    hook_to_n = {"1_hook": 1, "2_hooks": 2, "3_hooks": 3}
    for bs in batch_sizes:
        bs_data = data[data["param_batch_size"] == bs]
        points = []
        for hc in hook_configs:
            hc_data = bs_data[bs_data["param_hook_config"] == hc]
            if hc_data.empty:
                continue
            n = hook_to_n.get(hc, int(hc.split("_")[0]) if hc[0].isdigit() else 0)
            points.append((n, hc_data.iloc[0]["result_latency_ms_mean_ms"]))
        if not points:
            continue
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, "o-", label=f"batch={int(bs)}", linewidth=2, markersize=7)

    ax.set_xlabel("Number of active hook points")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Hook points ablation: linear scaling in hook count")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    _save(fig, output_dir / f"hook_points.{fmt}", f"hook_points.{fmt}")


# ── Config scaling ablation ───────────────────────────────────────────────────


def plot_config_scaling(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Latency vs max_steering_configs. Memory overlay if data is usable."""
    data = df[df["benchmark"] == "ablation.config_scaling"].copy()
    if data.empty:
        return

    data = data.sort_values("param_max_steering_configs")
    configs = data["param_max_steering_configs"].values
    lat_mean = data["result_latency_ms_mean_ms"].values
    lat_p10 = data.get(
        "result_latency_ms_p10_ms", pd.Series([np.nan] * len(data))
    ).values
    lat_p90 = data.get(
        "result_latency_ms_p90_ms", pd.Series([np.nan] * len(data))
    ).values

    # Memory overlay is only meaningful if values vary and aren't all zeros.
    # Early runs of bench_config_scaling.py used torch.cuda.memory_allocated()
    # which returns 0 for the vLLM subprocess; those runs record zeros that
    # would plot as a meaningless flat line. Skip the memory overlay in that
    # case. Also skip if all allocated_mb values are equal (a constant line
    # from KV cache dominating at gpu_memory_utilization=0.9).
    memory = data.get("result_allocated_mb")
    show_memory = False
    memory_vals = None
    if memory is not None and not memory.isna().all():
        memory_vals = memory.values
        nonzero = memory_vals[~np.isnan(memory_vals)]
        if len(nonzero) >= 2:
            spread = nonzero.max() - nonzero.min()
            # Require nonzero values and some meaningful variation (>1 MB)
            # so a flat line at ~20 GB from KV cache allocation gets skipped too.
            show_memory = bool((nonzero > 0).any() and spread > 1.0)

    fig, ax1 = plt.subplots()
    ax1.plot(
        configs, lat_mean, "o-", color="#2196F3", label="Mean latency", linewidth=2, markersize=7
    )
    if not np.isnan(lat_p10).all() and not np.isnan(lat_p90).all():
        ax1.fill_between(configs, lat_p10, lat_p90, alpha=0.2, color="#2196F3", label="p10–p90")
    ax1.set_xlabel("max_steering_configs")
    ax1.set_ylabel("Latency (ms)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_xscale("log", base=2)

    if show_memory:
        ax2 = ax1.twinx()
        ax2.plot(
            configs,
            memory_vals,
            "s--",
            color="#F44336",
            label="GPU memory",
            linewidth=2,
            markersize=7,
        )
        ax2.set_ylabel("Memory (MB)", color="#F44336")
        ax2.tick_params(axis="y", labelcolor="#F44336")
        title = "Config scaling: latency & memory vs max_steering_configs"
    else:
        title = "Config scaling: latency vs max_steering_configs"

    ax1.set_title(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    _save(fig, output_dir / f"config_scaling.{fmt}", f"config_scaling.{fmt}")


# ── Microbenchmarks ───────────────────────────────────────────────────────────


def plot_steering_op_microbench(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Steering op kernel latency vs num_tokens, grouped by hidden_size."""
    data = df[df["benchmark"] == "micro.steering_op"].copy()
    if data.empty:
        return

    fig, ax = plt.subplots()
    hidden_sizes = sorted(data["param_hidden_size"].dropna().unique())
    for h in hidden_sizes:
        h_data = data[data["param_hidden_size"] == h].sort_values("param_num_tokens")
        if h_data.empty:
            continue
        x = h_data["param_num_tokens"].values
        y = h_data["result_latency_ms_mean_ms"].values
        ax.plot(x, y, "o-", label=f"hidden={int(h)}", linewidth=2, markersize=6)

    ax.set_xlabel("num_tokens")
    ax.set_ylabel("Op latency (ms)")
    ax.set_title("Steering op kernel: latency vs num_tokens")
    ax.set_xscale("log", base=2)
    ax.legend()
    _save(fig, output_dir / f"microbench_op.{fmt}", f"microbench_op.{fmt}")


def plot_steering_manager_microbench(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """populate_steering_tables cost vs num_configs x num_layers."""
    data = df[df["benchmark"] == "micro.steering_manager"].copy()
    if data.empty:
        return

    fig, ax = plt.subplots()
    layer_counts = sorted(data["param_num_layers"].dropna().unique())
    for nl in layer_counts:
        nl_data = data[
            (data["param_num_layers"] == nl) & (data["param_hook_points"] == "3_hooks")
        ].sort_values("param_num_configs")
        if nl_data.empty:
            continue
        x = nl_data["param_num_configs"].values
        y = nl_data["result_populate_ms_mean_ms"].values
        ax.plot(x, y, "o-", label=f"{int(nl)} layers (3 hooks)", linewidth=2, markersize=6)

    ax.set_xlabel("num_configs")
    ax.set_ylabel("populate_steering_tables (ms)")
    ax.set_title("SteeringManager: populate cost scaling")
    ax.legend()
    _save(fig, output_dir / f"microbench_populate.{fmt}", f"microbench_populate.{fmt}")


# ── External libraries ────────────────────────────────────────────────────────


def plot_library_comparison(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """External library Tier 1 + Tier 2 comparison."""
    tier1 = df[df["benchmark"].str.startswith("external.tier1.", na=False)].copy()
    tier2 = df[df["benchmark"].str.startswith("external.tier2.", na=False)].copy()
    if tier1.empty and tier2.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if not tier1.empty:
        tier1 = tier1.sort_values("result_latency_ms_mean_ms")
        names = [b.replace("external.tier1.", "") for b in tier1["benchmark"]]
        means = tier1["result_latency_ms_mean_ms"].values
        colors = [COLORS.get(n, "#666") for n in names]
        axes[0].barh(names, means, color=colors)
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_title("Tier 1: single-request latency")
        for i, v in enumerate(means):
            if pd.notna(v):
                axes[0].text(v + max(means) * 0.02, i, f"{v:.0f}ms", va="center", fontsize=9)

    if not tier2.empty:
        tier2 = tier2.sort_values("result_req_per_sec", ascending=True)
        names = [b.replace("external.tier2.", "") for b in tier2["benchmark"]]
        rps = tier2["result_req_per_sec"].values
        colors = [COLORS.get(n, "#666") for n in names]
        axes[1].barh(names, rps, color=colors)
        axes[1].set_xlabel("Requests/sec")
        axes[1].set_title("Tier 2: batched throughput")
        for i, v in enumerate(rps):
            if pd.notna(v):
                axes[1].text(v + max(rps) * 0.02, i, f"{v:.1f}", va="center", fontsize=9)

    _save(fig, output_dir / f"library_comparison.{fmt}", f"library_comparison.{fmt}")


# ── Text summary ──────────────────────────────────────────────────────────────


def print_text_summary(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total records: {len(df)}")
    print(f"  Benchmarks present: {sorted(df['benchmark'].unique())}")
    print()

    # Latency headline
    latency = df[df["benchmark"] == "vllm.latency"]
    if not latency.empty:
        disabled = latency[latency["param_mode"] == "disabled"]
        per_req = latency[latency["param_mode"] == "per_request_1"]
        if not disabled.empty and not per_req.empty:
            print("  vLLM latency overhead (per_request_1 vs disabled):")
            for bs in sorted(disabled["param_batch_size"].dropna().unique()):
                d = disabled[disabled["param_batch_size"] == bs]
                p = per_req[per_req["param_batch_size"] == bs]
                if d.empty or p.empty:
                    continue
                d_ms = d.iloc[0]["result_latency_ms_mean_ms"]
                p_ms = p.iloc[0]["result_latency_ms_mean_ms"]
                if d_ms > 0:
                    ov = (p_ms - d_ms) / d_ms * 100
                    print(f"    batch={int(bs)}: {d_ms:.0f} → {p_ms:.0f} ms ({ov:+.1f}%)")

    # Memory
    memory = df[df["benchmark"] == "vllm.memory"]
    if not memory.empty and "result_steering_delta_mb" in memory.columns:
        print("\n  Memory cost per max_steering_configs (theoretical = bf16):")
        for _, row in memory.sort_values("param_max_steering_configs").iterrows():
            configs = row.get("param_max_steering_configs")
            delta = row.get("result_steering_delta_mb")
            model = row.get("param_model", "google/gemma-3-4b-it")
            if configs and delta is not None and configs > 0:
                theory = theoretical_memory_mb(model, int(configs))
                if theory is not None:
                    ratio = delta / theory if theory > 0 else 0
                    print(
                        f"    configs={int(configs)}: {delta:.1f} MB "
                        f"(theory: {theory:.1f} MB, ratio: {ratio:.2f}x)"
                    )
                else:
                    print(f"    configs={int(configs)}: {delta:.1f} MB")

    # Mixed batch
    mixed = df[df["benchmark"] == "vllm.mixed_batch"]
    if not mixed.empty:
        print("\n  Mixed-batch cost per active request (slope from num_active>0 points):")
        for bs in sorted(mixed["param_batch_size"].dropna().unique()):
            # Use only non-zero num_active points so the slope reflects the
            # marginal cost of an extra active request, not the discrete jump
            # from the disabled-mode baseline (which is dominated by table
            # population fixed cost).
            rows = (
                mixed[
                    (mixed["param_batch_size"] == bs)
                    & (mixed["param_num_active"] > 0)
                ]
                .sort_values("param_num_active")
            )
            if len(rows) < 2:
                continue
            xs = rows["param_num_active"].astype(float).to_numpy()
            ys = rows["result_latency_ms_mean_ms"].astype(float).to_numpy()
            # Linear least-squares slope (ms per additional active request)
            slope, intercept = np.polyfit(xs, ys, 1)
            n_lo = int(xs.min())
            n_hi = int(xs.max())
            print(
                f"    batch={int(bs)}: ~{slope:.1f} ms per additional active "
                f"request (linear fit over n={n_lo}..{n_hi}, "
                f"{len(rows)} points)"
            )

    # Throughput matrix
    tmat = df[df["benchmark"] == "vllm.throughput_matrix"]
    if not tmat.empty:
        print("\n  Throughput matrix loss vs disabled (mean over batch sizes):")
        for mode in ["enabled_idle", "mixed_25", "mixed_50", "mixed_75", "all_steered"]:
            mode_rows = tmat[tmat["param_mode"] == mode]
            if mode_rows.empty:
                continue
            losses = []
            for _, r in mode_rows.iterrows():
                bs = r.get("param_batch_size")
                base_row = tmat[
                    (tmat["param_mode"] == "disabled") & (tmat["param_batch_size"] == bs)
                ]
                if base_row.empty:
                    continue
                mode_tps = _tps_field(r)
                base_tps = _tps_field(base_row.iloc[0])
                if mode_tps and base_tps and base_tps > 0:
                    losses.append((base_tps - mode_tps) / base_tps * 100)
            if losses:
                avg = sum(losses) / len(losses)
                print(f"    {mode}: mean loss {avg:+.1f}% (n={len(losses)} batch sizes)")

    # CUDA graphs
    cuda = df[df["benchmark"] == "ablation.cuda_graphs"]
    if not cuda.empty:
        print("\n  CUDA graph interaction (speedup ratio with/without steering):")
        for bs in sorted(cuda["param_batch_size"].dropna().unique()):
            rows = cuda[cuda["param_batch_size"] == bs]
            try:
                gno = rows[
                    (rows["param_enforce_eager"] == False)  # noqa: E712
                    & (rows["param_enable_steering"] == False)
                ].iloc[0]["result_latency_ms_mean_ms"]
                gws = rows[
                    (rows["param_enforce_eager"] == False)
                    & (rows["param_enable_steering"] == True)
                ].iloc[0]["result_latency_ms_mean_ms"]
                eno = rows[
                    (rows["param_enforce_eager"] == True)
                    & (rows["param_enable_steering"] == False)
                ].iloc[0]["result_latency_ms_mean_ms"]
                ews = rows[
                    (rows["param_enforce_eager"] == True)
                    & (rows["param_enable_steering"] == True)
                ].iloc[0]["result_latency_ms_mean_ms"]
                ratio_nos = eno / gno
                ratio_wst = ews / gws
                per_step = (gws - gno) / 128
                print(
                    f"    batch={int(bs)}: no_steer {ratio_nos:.2f}x → "
                    f"w_steer {ratio_wst:.2f}x, per-step overhead ~{per_step:.1f}ms"
                )
            except (IndexError, KeyError):
                continue

    # Hook points
    hooks = df[df["benchmark"] == "ablation.hook_points"]
    if not hooks.empty:
        print("\n  Hook points scaling:")
        for bs in sorted(hooks["param_batch_size"].dropna().unique()):
            rows = hooks[hooks["param_batch_size"] == bs]
            hc_to_lat = {
                r["param_hook_config"]: r["result_latency_ms_mean_ms"]
                for _, r in rows.iterrows()
            }
            if "1_hook" in hc_to_lat and "3_hooks" in hc_to_lat:
                h1 = hc_to_lat["1_hook"]
                h3 = hc_to_lat["3_hooks"]
                if h1 > 0:
                    pct = (h3 - h1) / h1 * 100
                    print(f"    batch={int(bs)}: 1→3 hooks adds {h3 - h1:.0f}ms ({pct:+.1f}%)")

    # Table sizing interaction
    tsize = df[df["benchmark"] == "vllm.table_sizing"]
    if not tsize.empty:
        # Build disabled baseline lookup
        base = tsize[tsize["param_mode"] == "disabled"]
        base_tps: dict[float, float] = {}
        for _, r in base.iterrows():
            bs = r.get("param_batch_size")
            tps = _tps_field(r)
            if tps and pd.notna(bs):
                base_tps[bs] = tps

        steered = tsize[tsize["param_mode"] == "steered"]
        if not steered.empty and base_tps:
            max_cfgs = sorted(steered["param_max_steering_configs"].dropna().unique())
            if len(max_cfgs) >= 2:
                small_max = min(max_cfgs)
                large_max = max(max_cfgs)
                print(
                    f"\n  Table sizing benefit (max_cfg={int(large_max)} vs "
                    f"{int(small_max)}) — throughput loss reduction:"
                )
                for bs in sorted(steered["param_batch_size"].dropna().unique()):
                    for distinct in sorted(
                        steered["param_distinct_configs"].dropna().unique()
                    ):
                        small_row = steered[
                            (steered["param_batch_size"] == bs)
                            & (steered["param_max_steering_configs"] == small_max)
                            & (steered["param_distinct_configs"] == distinct)
                        ]
                        large_row = steered[
                            (steered["param_batch_size"] == bs)
                            & (steered["param_max_steering_configs"] == large_max)
                            & (steered["param_distinct_configs"] == distinct)
                        ]
                        baseline = base_tps.get(bs)
                        if (
                            small_row.empty
                            or large_row.empty
                            or baseline is None
                            or baseline <= 0
                        ):
                            continue
                        small_tps = _tps_field(small_row.iloc[0])
                        large_tps = _tps_field(large_row.iloc[0])
                        if small_tps is None or large_tps is None:
                            continue
                        small_loss = (baseline - small_tps) / baseline * 100
                        large_loss = (baseline - large_tps) / baseline * 100
                        delta = small_loss - large_loss
                        print(
                            f"    batch={int(bs)} distinct={int(distinct)}: "
                            f"{small_loss:.1f}% → {large_loss:.1f}% "
                            f"(benefit: {delta:+.1f}pp)"
                        )

    print(f"\n{'=' * 70}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot benchmark results")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="results/plots/")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--tag", default="", help="Filter results by tag")
    args = parser.parse_args()

    setup_style()

    print(f"Loading results from {args.results_dir}...")
    df = aggregate(args.results_dir)
    if df.empty:
        print("No results found. Run benchmarks first.")
        sys.exit(1)

    if args.tag:
        df = df[df["tag"] == args.tag]

    print(f"Loaded {len(df)} result records")
    print(f"Benchmarks: {sorted(df['benchmark'].unique())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots...")

    # vLLM system benchmarks
    plot_overhead_bars(df, output_dir, args.format)
    plot_throughput_by_configs(df, output_dir, args.format)
    plot_throughput_matrix(df, output_dir, args.format)
    plot_mixed_batch(df, output_dir, args.format)
    plot_max_tokens_sweep(df, output_dir, args.format)
    plot_table_sizing(df, output_dir, args.format)
    plot_memory_scaling(df, output_dir, args.format)

    # Ablations
    plot_cuda_graphs_ablation(df, output_dir, args.format)
    plot_hook_points(df, output_dir, args.format)
    plot_config_scaling(df, output_dir, args.format)

    # Microbenchmarks
    plot_steering_op_microbench(df, output_dir, args.format)
    plot_steering_manager_microbench(df, output_dir, args.format)

    # External comparison
    plot_library_comparison(df, output_dir, args.format)

    # Text summary
    print_text_summary(df)

    # Export aggregated CSV for manual exploration
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results CSV: {csv_path}")
    print(f"Plots written to: {args.output_dir}")


if __name__ == "__main__":
    main()
