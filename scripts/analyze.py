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
    """Throughput matrix benchmark: 4 plots covering latency and throughput."""
    data = df[df["benchmark"] == "vllm.throughput_matrix"].copy()
    if data.empty:
        return

    mode_order = [
        "disabled",
        "enabled_idle",
        "mixed_25",
        "mixed_50",
        "mixed_75",
        "all_steered",
    ]
    batch_sizes = sorted(data["param_batch_size"].dropna().unique())
    if not batch_sizes:
        return

    modes_present = [m for m in mode_order if not data[data["param_mode"] == m].empty]

    # 1. Throughput grouped bars
    _grouped_bars(
        data,
        modes_present,
        batch_sizes,
        metric="throughput",
        ylabel="Throughput (tokens/sec)",
        title="Throughput matrix: mode × batch_size",
        output_path=output_dir / f"matrix_throughput_bars.{fmt}",
        name=f"matrix_throughput_bars.{fmt}",
    )

    # 2. Throughput loss heatmap
    _loss_heatmap(
        data,
        modes_present,
        batch_sizes,
        metric="throughput",
        label="throughput loss (%)",
        title="Throughput loss (%) vs disabled",
        output_path=output_dir / f"matrix_throughput_heatmap.{fmt}",
        name=f"matrix_throughput_heatmap.{fmt}",
    )

    # 3. Latency grouped bars
    _grouped_bars(
        data,
        modes_present,
        batch_sizes,
        metric="latency",
        ylabel="Latency (ms)",
        title="Latency matrix: mode × batch_size",
        output_path=output_dir / f"matrix_latency_bars.{fmt}",
        name=f"matrix_latency_bars.{fmt}",
    )

    # 4. Latency overhead heatmap
    _loss_heatmap(
        data,
        modes_present,
        batch_sizes,
        metric="latency",
        label="latency overhead (%)",
        title="Latency overhead (%) vs disabled",
        output_path=output_dir / f"matrix_latency_heatmap.{fmt}",
        name=f"matrix_latency_heatmap.{fmt}",
    )


# ── Mixed batch (proportional scaling) ────────────────────────────────────────


def plot_mixed_batch(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Mixed batch: latency and throughput vs num_active within a fixed batch size."""
    data = df[df["benchmark"] == "vllm.mixed_batch"].copy()
    if data.empty:
        return

    # Group by batch_size; one line per batch size
    batch_sizes = sorted(data["param_batch_size"].dropna().unique())
    if not batch_sizes:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for bs in batch_sizes:
        bs_data = data[data["param_batch_size"] == bs].sort_values("param_num_active")
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

    axes[1].set_xlabel("Number of steered requests in batch")
    axes[1].set_ylabel("Throughput (tokens/sec)")
    axes[1].set_title("Mixed-batch throughput: proportional to active count")
    axes[1].legend()

    _save(fig, output_dir / f"mixed_batch.{fmt}", f"mixed_batch.{fmt}")


# ── Memory ────────────────────────────────────────────────────────────────────


def plot_memory_scaling(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Memory cost vs max_steering_configs, measured vs theoretical."""
    data = df[df["benchmark"] == "vllm.memory"].copy()
    if data.empty:
        return

    data = data.sort_values("param_max_steering_configs")
    # Filter to rows that have a delta (skip the baseline row if null)
    configs = data["param_max_steering_configs"].values
    delta = data.get("result_steering_delta_mb")
    theoretical = data.get("result_theoretical_mb")

    if delta is None:
        return

    fig, ax = plt.subplots()
    ax.plot(configs, delta.values, "o-", label="Measured", linewidth=2, markersize=7, color="#2196F3")
    if theoretical is not None and not theoretical.isna().all():
        ax.plot(
            configs,
            theoretical.values,
            "s--",
            label="Theoretical (bf16)",
            linewidth=2,
            markersize=7,
            color="#FF9800",
        )
    ax.set_xlabel("max_steering_configs")
    ax.set_ylabel("Steering buffer memory (MB)")
    ax.set_title("Memory scaling: measured vs theoretical (Gemma-3-4B)")
    ax.legend()
    _save(fig, output_dir / f"memory_scaling.{fmt}", f"memory_scaling.{fmt}")


# ── CUDA graphs ablation ──────────────────────────────────────────────────────


def plot_cuda_graphs_ablation(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """CUDA graphs 2x2 heatmap and per-step overhead derivation."""
    data = df[df["benchmark"] == "ablation.cuda_graphs"].copy()
    if data.empty:
        return

    batch_sizes = sorted(data["param_batch_size"].dropna().unique())

    # 2x2 heatmap per batch size
    for bs in batch_sizes:
        bs_data = data[data["param_batch_size"] == bs]
        if len(bs_data) < 4:
            continue
        matrix = np.zeros((2, 2))
        for _, row in bs_data.iterrows():
            eager_idx = 1 if row.get("param_enforce_eager") else 0
            steer_idx = 1 if row.get("param_enable_steering") else 0
            matrix[eager_idx, steer_idx] = row.get("result_latency_ms_mean_ms", 0)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Steering Off", "Steering On"])
        ax.set_yticklabels(["CUDA Graphs", "Eager"])
        ax.set_title(f"CUDA Graphs × Steering (batch={int(bs)})")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.0f}ms",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )
        fig.colorbar(im, label="Latency (ms)")
        _save(
            fig,
            output_dir / f"cuda_graphs_bs{int(bs)}.{fmt}",
            f"cuda_graphs_bs{int(bs)}.{fmt}",
        )

    # Per-step overhead derivation (the headline fusion-loss chart)
    batch_sizes_sorted = sorted(batch_sizes)
    graphs_deltas = []
    eager_deltas = []
    for bs in batch_sizes_sorted:
        rows = data[data["param_batch_size"] == bs]
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
        except (IndexError, KeyError):
            continue
        # Assume ~128 decode steps
        steps = 128
        graphs_deltas.append((bs, (gws - gno) / steps))
        eager_deltas.append((bs, (ews - eno) / steps))

    if graphs_deltas and eager_deltas:
        fig, ax = plt.subplots()
        gbs, gvals = zip(*graphs_deltas)
        ebs, evals = zip(*eager_deltas)
        ax.plot(gbs, gvals, "o-", label="with CUDA graphs", linewidth=2, markersize=7)
        ax.plot(ebs, evals, "s--", label="eager (no graphs)", linewidth=2, markersize=7)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Per-step steering overhead (ms)")
        ax.set_title("Per-step steering overhead scales with batch size")
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
    """Latency and memory vs max_steering_configs."""
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
    memory = data.get("result_allocated_mb")

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

    if memory is not None and not memory.isna().all():
        ax2 = ax1.twinx()
        ax2.plot(
            configs,
            memory.values,
            "s--",
            color="#F44336",
            label="GPU memory",
            linewidth=2,
            markersize=7,
        )
        ax2.set_ylabel("Memory (MB)", color="#F44336")
        ax2.tick_params(axis="y", labelcolor="#F44336")

    ax1.set_title("Config scaling: latency & memory vs max_steering_configs")
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
        print("\n  Memory cost per max_steering_configs:")
        for _, row in memory.sort_values("param_max_steering_configs").iterrows():
            configs = row.get("param_max_steering_configs")
            delta = row.get("result_steering_delta_mb")
            theory = row.get("result_theoretical_mb")
            if configs and delta is not None and configs > 0:
                theory_str = f" (theory: {theory:.1f} MB)" if theory else ""
                print(f"    configs={int(configs)}: {delta:.1f} MB{theory_str}")

    # Mixed batch
    mixed = df[df["benchmark"] == "vllm.mixed_batch"]
    if not mixed.empty:
        print("\n  Mixed-batch cost per active request:")
        for bs in sorted(mixed["param_batch_size"].dropna().unique()):
            rows = mixed[mixed["param_batch_size"] == bs].sort_values("param_num_active")
            if len(rows) < 2:
                continue
            first = rows.iloc[0]
            last = rows.iloc[-1]
            delta_n = last["param_num_active"] - first["param_num_active"]
            delta_lat = (
                last["result_latency_ms_mean_ms"] - first["result_latency_ms_mean_ms"]
            )
            if delta_n > 0:
                per_req_cost = delta_lat / delta_n
                print(
                    f"    batch={int(bs)}: ~{per_req_cost:.1f} ms per additional "
                    f"active request (n={int(first['param_num_active'])} → "
                    f"{int(last['param_num_active'])})"
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
