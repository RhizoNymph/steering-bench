"""JSON result schema and environment capture.

Every benchmark writes a JSON file following this schema so that
analysis scripts can aggregate results uniformly.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def capture_environment() -> dict[str, Any]:
    """Capture GPU, software, and host environment metadata."""
    env: dict[str, Any] = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "platform": platform.platform(),
    }

    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        try:
            uuid_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip()
            env["gpu_uuid"] = uuid_out.split("\n")[0]
        except (subprocess.SubprocessError, FileNotFoundError):
            env["gpu_uuid"] = "unknown"

        try:
            clocks_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.current.graphics,clocks.max.graphics",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            parts = clocks_out.split("\n")[0].split(", ")
            env["gpu_clock_current_mhz"] = int(parts[0])
            env["gpu_clock_max_mhz"] = int(parts[1])
            env["clocks_pinned"] = abs(int(parts[0]) - int(parts[1])) < 50
        except (subprocess.SubprocessError, FileNotFoundError, (ValueError, IndexError)):
            env["clocks_pinned"] = False
    else:
        env["gpu"] = "none"

    # vLLM version if installed
    try:
        import vllm
        env["vllm_version"] = vllm.__version__
        # Try to get commit hash
        try:
            env["vllm_commit"] = vllm.__commit__  # type: ignore[attr-defined]
        except AttributeError:
            pass
    except ImportError:
        env["vllm_version"] = "not installed"

    return env


def write_result(
    benchmark: str,
    parameters: dict[str, Any],
    results: dict[str, Any],
    output_dir: str | Path,
    tag: str = "",
    raw_samples_ms: list[float] | None = None,
) -> Path:
    """Write a benchmark result to a JSON file.

    Args:
        benchmark: Benchmark identifier (e.g. "micro.steering_op").
        parameters: Benchmark parameters used.
        results: Measured results (latency, throughput, memory, etc.).
        output_dir: Directory to write the JSON file.
        tag: Optional tag for this run.
        raw_samples_ms: Optional raw timing samples.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    safe_name = benchmark.replace(".", "_").replace("/", "_")

    record = {
        "benchmark": benchmark,
        "timestamp": timestamp,
        "tag": tag,
        "environment": capture_environment(),
        "parameters": parameters,
        "results": results,
    }
    if raw_samples_ms is not None:
        record["raw_samples_ms"] = raw_samples_ms

    filename = f"{safe_name}_{timestamp.replace(':', '-').replace('+', '_')}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return path


def print_result_summary(benchmark: str, results: dict[str, Any]) -> None:
    """Print a human-readable summary of benchmark results to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {benchmark}")
    print(f"{'=' * 60}")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")
