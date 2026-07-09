"""Patch-sweep registry / discovery + harness wiring.

Runs WITHOUT transformer_lens / httpx / a live server by monkeypatching the
package-availability probe.
"""

from __future__ import annotations

from steering_bench.engine import patch_sweep as ps
from steering_bench.engine.base import Capabilities


def test_registry_contains_both_adapters() -> None:
    names = {e.name for e in ps.PATCH_SWEEP_REGISTRY}
    assert names == {"tl", "vllm"}


def test_registry_required_packages_and_base_url() -> None:
    by_name = {e.name: e for e in ps.PATCH_SWEEP_REGISTRY}
    assert by_name["tl"].required_package == "transformer_lens"
    assert by_name["tl"].needs_base_url is False
    # vLLM adapter is always importable; availability is a reachable server.
    assert by_name["vllm"].required_package is None
    assert by_name["vllm"].needs_base_url is True


def test_discover_lists_both_when_available(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ps, "is_package_available", lambda pkg: True)
    classes = ps.discover_patch_sweep()
    names = {c.name for c in classes}
    assert names == {"tl", "vllm"}
    out = capsys.readouterr().out
    assert "tl: available" in out
    assert "vllm: available" in out
    # vLLM's annotation warns a server is required.
    assert "requires a reachable --base-url server" in out


def test_discover_skips_tl_when_transformer_lens_absent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ps, "is_package_available", lambda pkg: pkg != "transformer_lens"
    )
    classes = ps.discover_patch_sweep()
    names = {c.name for c in classes}
    assert "tl" not in names
    # vLLM has no package dependency -> still listed.
    assert "vllm" in names
    out = capsys.readouterr().out
    assert "tl: SKIPPED (transformer_lens not installed)" in out


def test_discover_filter_by_name(monkeypatch) -> None:
    monkeypatch.setattr(ps, "is_package_available", lambda pkg: True)
    classes = ps.discover_patch_sweep(filter_names=["vllm"])
    assert [c.name for c in classes] == ["vllm"]


def test_get_patch_sweep_engine_unknown() -> None:
    import pytest

    with pytest.raises(ps.PatchSweepError):
        ps.get_patch_sweep_engine("nope")


def test_get_patch_sweep_engine_imports_adapters() -> None:
    tl = ps.get_patch_sweep_engine("tl")
    vllm = ps.get_patch_sweep_engine("vllm")
    assert tl.name == "tl"
    assert vllm.name == "vllm"
    assert issubclass(tl, ps.PatchSweepEngine)
    assert issubclass(vllm, ps.PatchSweepEngine)
    # Both advertise the patch_sweep capability.
    assert tl.capabilities.satisfies(Capabilities(patch_sweep=True))
    assert vllm.capabilities.satisfies(Capabilities(patch_sweep=True))


def test_is_package_available_none_is_true() -> None:
    assert ps.is_package_available(None) is True


def test_harness_registry_and_variant_parsing() -> None:
    from steering_bench.harness.benchmarks.patch_sweep import (
        PatchSweepBenchmark,
        PatchSweepModeError,
        parse_variants,
    )
    from steering_bench.harness.benchmarks.registry import (
        PATCH_SWEEP_REGISTRY,
        get_patch_sweep_benchmark,
    )

    assert "patch-sweep" in PATCH_SWEEP_REGISTRY
    assert get_patch_sweep_benchmark("patch-sweep") is PatchSweepBenchmark
    # Not a sync Benchmark subclass (its own axis).
    from steering_bench.harness.benchmark import Benchmark

    assert not issubclass(PatchSweepBenchmark, Benchmark)

    tl_variants, want_vllm = parse_variants(["tl_naive", "tl_batched", "vllm_sweep"])
    assert tl_variants == ["naive", "batched"]
    assert want_vllm is True
    tl_only, no_vllm = parse_variants(["tl_batched"])
    assert tl_only == ["batched"] and no_vllm is False

    import pytest

    with pytest.raises(PatchSweepModeError):
        parse_variants(["bogus"])
