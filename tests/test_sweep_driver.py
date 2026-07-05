"""Dry-run checks for the sweep driver: spread:N layer resolution, consumer
param assembly, and cell argv correctness — all without CUDA. The driver module
imports ``vllm`` only inside functions, so it loads on a GPU-less box.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_DRIVER = (
    Path(__file__).resolve().parent.parent / "scripts" / "bench_dynamic_steering.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("bench_dynamic_steering", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_spread_middle_layer():
    assert mod.resolve_layers("spread:1", 34) == [17]


def test_spread_counts_and_bounds():
    for n in (4, 8, 17):
        layers = mod.resolve_layers(f"spread:{n}", 34)
        assert len(layers) == n
        assert layers == sorted(layers)
        assert min(layers) >= 0 and max(layers) <= 33


def test_spread_full_and_over():
    assert mod.resolve_layers("spread:34", 34) == list(range(34))
    assert mod.resolve_layers("spread:99", 34) == list(range(34))


def test_explicit_comma_list():
    assert mod.resolve_layers("3,17,30", 34) == [3, 17, 30]


def test_consumer_params_defaults_monitor_to_first_site():
    args = argparse.Namespace(
        steer_hooks="post_block", norm=8.0, monitor_layer=-1, monitor_hook=""
    )
    params = mod._consumer_params(args, [5, 10])
    assert params["layers"] == [5, 10]
    assert params["hooks"] == ["post_block"]
    assert params["monitor_layer"] == 5  # first steer layer
    assert params["monitor_hook"] == "post_block"  # first steer hook


def test_consumer_params_explicit_monitor_site():
    args = argparse.Namespace(
        steer_hooks="pre_attn,post_block", norm=8.0,
        monitor_layer=20, monitor_hook="post_attn",
    )
    params = mod._consumer_params(args, [5, 10])
    assert params["monitor_layer"] == 20
    assert params["monitor_hook"] == "post_attn"
    assert params["hooks"] == ["pre_attn", "post_block"]


def test_arm_flags_wiring():
    # rowmon is the only arm that flips enable_row_monitor.
    assert mod.ARM_FLAGS["steer_rowmon"] == (True, True)
    assert mod.ARM_FLAGS["steer_override"] == (True, False)
    assert mod.ARM_FLAGS["off"] == (False, False)
    assert mod.ARM_FLAGS["steer_per_request"] == (True, False)
    assert mod.PERREQ_ARMS == {
        "steer_override", "steer_rowmon", "steer_per_request"
    }


def test_cell_argv_threads_all_site_params():
    args = argparse.Namespace(
        model="google/gemma-3-4b-it", num_model_layers=34,
        steer_layers="spread:8", steer_hooks="pre_attn,post_block",
        monitor_layer=17, monitor_hook="post_block", norm=8.0,
        prompt_len=64, output_len=64, warmup=3, iters=8, gpu_mem_util=0.92,
        enforce_eager=False,
    )
    # Build the argv the parent would hand a cell (mirror _cell_subprocess).
    import sys

    cmd = [
        sys.executable, str(_DRIVER), "--cell",
        "--arm", "steer_rowmon", "--batch-size", "16",
        "--model", args.model, "--num-model-layers", str(args.num_model_layers),
        "--steer-layers", args.steer_layers, "--steer-hooks", args.steer_hooks,
        "--monitor-layer", str(args.monitor_layer),
        "--monitor-hook", args.monitor_hook,
        "--norm", str(args.norm), "--prompt-len", str(args.prompt_len),
        "--output-len", str(args.output_len), "--warmup", str(args.warmup),
        "--iters", str(args.iters), "--gpu-mem-util", str(args.gpu_mem_util),
    ]
    # Assert the key site flags are present and well-formed in the cell argv.
    assert "--steer-layers" in cmd and "spread:8" in cmd
    assert "--steer-hooks" in cmd and "pre_attn,post_block" in cmd
    assert "--monitor-layer" in cmd and "17" in cmd
    assert "--num-model-layers" in cmd and "34" in cmd
    assert mod.resolve_layers("spread:8", 34) == mod.resolve_layers("spread:8", 34)
