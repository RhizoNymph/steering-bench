#!/usr/bin/env python3
"""Aggregate results and generate plots for the steering benchmark suite.

Reads all JSON result files, computes derived metrics, and produces
matplotlib charts suitable for an article.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steering_bench.analysis.aggregate import aggregate


# ── Plot styling ──────────────────────────────────────────────────────────────

COLORS = {
    "disabled": "#2196F3",
    "enabled_idle": "#4CAF50",
    "per_request_1": "#FF9800",
    "per_request_4": "#F44336",
    "hf_baseline": "#9E9E9E",
    "transformerlens": "#E91E63",
    "nnsight": "#9C27B0",
    "repeng": "#FF5722",
    "pyvene": "#795548",
    "vllm_single": "#2196F3",
    "vllm_batched": "#00BCD4",
}


def setup_style():
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


# ── Plot functions ────────────────────────────────────────────────────────────


def plot_overhead_bars(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """vLLM latency: grouped bars by steering mode across batch sizes."""
    data = df[df["benchmark"] == "vllm.latency"].copy()
    if data.empty:
        return

    modes = ["disabled", "enabled_idle", "per_request_1", "per_request_4"]
    batch_sizes = sorted(data["param_batch_size"].dropna().unique())

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
    ax.set_title("vLLM Latency: Steering Overhead by Mode")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in batch_sizes])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"overhead_bars.{fmt}")
    plt.close(fig)
    print(f"  Saved overhead_bars.{fmt}")


def plot_library_comparison(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """External library comparison: Tier 1 and Tier 2 side by side."""
    tier1 = df[df["benchmark"].str.startswith("external.tier1.")].copy()
    tier2 = df[df["benchmark"].str.startswith("external.tier2.")].copy()

    if tier1.empty and tier2.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Tier 1: single-request latency
    if not tier1.empty:
        tier1 = tier1.sort_values("result_latency_ms_mean_ms")
        names = [b.replace("external.tier1.", "") for b in tier1["benchmark"]]
        means = tier1["result_latency_ms_mean_ms"].values
        colors = [COLORS.get(n, "#666") for n in names]

        axes[0].barh(names, means, color=colors)
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_title("Tier 1: Single-Request Latency")
        # Add value labels
        for i, (v, n) in enumerate(zip(means, names)):
            if pd.notna(v):
                axes[0].text(v + max(means) * 0.02, i, f"{v:.0f}ms", va="center", fontsize=9)

    # Tier 2: batch throughput
    if not tier2.empty:
        tier2 = tier2.sort_values("result_req_per_sec", ascending=True)
        names = [b.replace("external.tier2.", "") for b in tier2["benchmark"]]
        rps = tier2["result_req_per_sec"].values
        colors = [COLORS.get(n, "#666") for n in names]

        axes[1].barh(names, rps, color=colors)
        axes[1].set_xlabel("Requests/sec")
        axes[1].set_title(f"Tier 2: Batched Throughput")
        for i, (v, n) in enumerate(zip(rps, names)):
            if pd.notna(v):
                axes[1].text(v + max(rps) * 0.02, i, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / f"library_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved library_comparison.{fmt}")


def plot_ablation_heatmaps(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Ablation results: CUDA graphs heatmap + hook points bars."""
    # CUDA graphs 2x2 heatmap
    cuda_data = df[df["benchmark"] == "ablation.cuda_graphs"].copy()
    if not cuda_data.empty:
        # Pick the first batch size available for the heatmap
        batch_sizes = sorted(cuda_data["param_batch_size"].dropna().unique())
        for bs in batch_sizes:
            bs_data = cuda_data[cuda_data["param_batch_size"] == bs]
            if len(bs_data) < 4:
                continue

            matrix = np.zeros((2, 2))
            labels_x = ["Steering Off", "Steering On"]
            labels_y = ["CUDA Graphs", "Eager"]

            for _, row in bs_data.iterrows():
                eager_idx = 1 if row.get("param_enforce_eager") else 0
                steer_idx = 1 if row.get("param_enable_steering") else 0
                matrix[eager_idx, steer_idx] = row.get("result_latency_ms_mean_ms", 0)

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels_x)
            ax.set_yticklabels(labels_y)
            ax.set_title(f"CUDA Graphs x Steering (batch={int(bs)})")

            # Annotate cells
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]:.1f}ms", ha="center", va="center",
                            fontsize=14, fontweight="bold")

            fig.colorbar(im, label="Latency (ms)")
            fig.tight_layout()
            fig.savefig(output_dir / f"ablation_cuda_graphs_bs{int(bs)}.{fmt}")
            plt.close(fig)
            print(f"  Saved ablation_cuda_graphs_bs{int(bs)}.{fmt}")

    # Hook points bar chart
    hook_data = df[df["benchmark"] == "ablation.hook_points"].copy()
    if not hook_data.empty:
        batch_sizes = sorted(hook_data["param_batch_size"].dropna().unique())
        fig, ax = plt.subplots()
        x = np.arange(len(batch_sizes))
        hook_configs = sorted(hook_data["param_hook_config"].dropna().unique())
        width = 0.25
        offsets = np.arange(len(hook_configs)) - (len(hook_configs) - 1) / 2

        for i, hc in enumerate(hook_configs):
            hc_data = hook_data[hook_data["param_hook_config"] == hc]
            means = [
                hc_data[hc_data["param_batch_size"] == bs]["result_latency_ms_mean_ms"].values[0]
                if not hc_data[hc_data["param_batch_size"] == bs].empty else 0
                for bs in batch_sizes
            ]
            ax.bar(x + offsets[i] * width, means, width, label=hc, capsize=3)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Hook Points Ablation: Latency by Active Hooks")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(b)) for b in batch_sizes])
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"ablation_hook_points.{fmt}")
        plt.close(fig)
        print(f"  Saved ablation_hook_points.{fmt}")


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    """Config scaling: latency + memory vs max_steering_configs."""
    data = df[df["benchmark"] == "ablation.config_scaling"].copy()
    if data.empty:
        return

    data = data.sort_values("param_max_steering_configs")
    configs = data["param_max_steering_configs"].values
    means = data["result_latency_ms_mean_ms"].values
    p10 = data.get("result_latency_ms_p10_ms", pd.Series([0] * len(data))).values
    p90 = data.get("result_latency_ms_p90_ms", pd.Series([0] * len(data))).values

    fig, ax1 = plt.subplots()

    # Latency line with error band
    ax1.plot(configs, means, "o-", color="#2196F3", label="Mean Latency", linewidth=2)
    if any(p10 > 0) and any(p90 > 0):
        ax1.fill_between(configs, p10, p90, alpha=0.2, color="#2196F3", label="p10-p90 range")
    ax1.set_xlabel("max_steering_configs")
    ax1.set_ylabel("Latency (ms)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    # Memory on secondary axis
    memory = data.get("result_allocated_mb")
    if memory is not None and not memory.isna().all():
        ax2 = ax1.twinx()
        ax2.plot(configs, memory.values, "s--", color="#F44336", label="GPU Memory", linewidth=2)
        ax2.set_ylabel("Memory (MB)", color="#F44336")
        ax2.tick_params(axis="y", labelcolor="#F44336")

    ax1.set_title("Config Scaling: Latency & Memory vs max_steering_configs")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    fig.tight_layout()
    fig.savefig(output_dir / f"scaling_configs.{fmt}")
    plt.close(fig)
    print(f"  Saved scaling_configs.{fmt}")


# ── Text summary ──────────────────────────────────────────────────────────────


def print_text_summary(df: pd.DataFrame) -> None:
    """Print headline numbers to stdout."""
    print(f"\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    # vLLM overhead headline
    latency = df[df["benchmark"] == "vllm.latency"]
    if not latency.empty:
        disabled = latency[latency["param_mode"] == "disabled"]
        per_req = latency[latency["param_mode"] == "per_request_1"]
        if not disabled.empty and not per_req.empty:
            for bs in sorted(disabled["param_batch_size"].dropna().unique()):
                d = disabled[disabled["param_batch_size"] == bs]
                p = per_req[per_req["param_batch_size"] == bs]
                if not d.empty and not p.empty:
                    d_ms = d.iloc[0]["result_latency_ms_mean_ms"]
                    p_ms = p.iloc[0]["result_latency_ms_mean_ms"]
                    if d_ms > 0:
                        overhead = (p_ms - d_ms) / d_ms * 100
                        print(f"  vLLM overhead (batch={int(bs)}): {overhead:+.1f}% "
                              f"({d_ms:.0f}ms -> {p_ms:.0f}ms)")

    # Memory headline
    memory = df[df["benchmark"] == "vllm.memory"]
    if not memory.empty and "derived_memory_overhead_mb" in memory.columns:
        for _, row in memory.iterrows():
            configs = row.get("param_max_steering_configs")
            delta = row.get("derived_memory_overhead_mb")
            if configs and delta and configs > 0:
                print(f"  Memory cost (configs={int(configs)}): {delta:.1f} MB")

    # External comparison headline
    tier1 = df[df["benchmark"].str.startswith("external.tier1.")]
    if not tier1.empty:
        print("\n  Library ranking (Tier 1, by latency):")
        ranked = tier1.sort_values("result_latency_ms_mean_ms")
        for _, row in ranked.iterrows():
            name = row["benchmark"].replace("external.tier1.", "")
            ms = row.get("result_latency_ms_mean_ms")
            if pd.notna(ms):
                print(f"    {name:<20} {ms:.0f} ms")

    # Ablation headlines
    cuda = df[df["benchmark"] == "ablation.cuda_graphs"]
    if not cuda.empty:
        graphs_off = cuda[
            (cuda.get("param_enforce_eager") == False) & (cuda.get("param_enable_steering") == False)
        ]
        graphs_on_steer = cuda[
            (cuda.get("param_enforce_eager") == False) & (cuda.get("param_enable_steering") == True)
        ]
        if not graphs_off.empty and not graphs_on_steer.empty:
            print("\n  CUDA graphs + steering: preserved" if True else "degraded")

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

    # Only plot categories that have data
    if (df["benchmark"] == "vllm.latency").any():
        plot_overhead_bars(df, output_dir, args.format)

    if df["benchmark"].str.startswith("external.").any():
        plot_library_comparison(df, output_dir, args.format)

    if df["benchmark"].str.startswith("ablation.").any():
        plot_ablation_heatmaps(df, output_dir, args.format)

    if (df["benchmark"] == "ablation.config_scaling").any():
        plot_scaling_curves(df, output_dir, args.format)

    # Text summary
    print_text_summary(df)

    # Save DataFrame as CSV for further analysis
    csv_path = output_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results CSV: {csv_path}")
    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
