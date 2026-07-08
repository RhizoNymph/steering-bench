"""Registry / capability tests for the engine seam.

These run WITHOUT a GPU and WITHOUT any engine package installed by
monkeypatching the package-availability probe.
"""

from __future__ import annotations

from steering_bench.engine import base as base_mod
from steering_bench.engine import registry as reg
from steering_bench.engine.base import Capabilities


def test_satisfies_all_dont_care() -> None:
    # An all-False requirement is satisfied by everything.
    empty = Capabilities()
    assert Capabilities().satisfies(empty)
    assert Capabilities(batching=True).satisfies(empty)
    assert Capabilities(batching=True, capture=True).satisfies(empty)


def test_satisfies_requires_present() -> None:
    full = Capabilities(
        batching=True,
        named_modules=True,
        multi_layer=True,
        multi_hook=True,
        capture=True,
    )
    assert full.satisfies(Capabilities(batching=True))
    assert full.satisfies(Capabilities(capture=True, multi_hook=True))
    assert full.satisfies(full)


def test_satisfies_missing_fails() -> None:
    subset = Capabilities(batching=False, named_modules=False)
    assert not subset.satisfies(Capabilities(batching=True))
    assert not subset.satisfies(Capabilities(named_modules=True))
    # Providing an unrelated capability does not help.
    assert not Capabilities(capture=True).satisfies(Capabilities(batching=True))


def test_satisfies_matrix() -> None:
    for req_batch in (False, True):
        for prov_batch in (False, True):
            req = Capabilities(batching=req_batch)
            prov = Capabilities(batching=prov_batch)
            # required=False -> always ok; required=True -> needs provided True.
            expected = (not req_batch) or prov_batch
            assert prov.satisfies(req) is expected


_ALL_ENGINES = {"vllm", "transformerlens", "hf_baseline", "nnsight", "repeng", "pyvene"}


def test_registry_contains_known_engines() -> None:
    names = {e.name for e in reg.ENGINE_REGISTRY}
    assert _ALL_ENGINES <= names


def test_registry_required_packages() -> None:
    by_name = {e.name: e for e in reg.ENGINE_REGISTRY}
    assert by_name["hf_baseline"].required_package == "transformers"
    assert by_name["nnsight"].required_package == "nnsight"
    assert by_name["repeng"].required_package == "repeng"
    assert by_name["pyvene"].required_package == "pyvene"


def test_discover_all_available(monkeypatch, capsys) -> None:
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: True)
    classes = reg.discover()
    class_names = {c.name for c in classes}
    assert _ALL_ENGINES <= class_names
    out = capsys.readouterr().out
    assert "vllm: available" in out
    for name in ("hf_baseline", "nnsight", "repeng", "pyvene"):
        assert f"{name}: available" in out


def test_discover_skips_new_engines_when_package_absent(monkeypatch, capsys) -> None:
    # Only transformers present; nnsight/repeng/pyvene absent.
    present = {"transformers", "transformer_lens", "vllm"}
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: pkg in present)
    classes = reg.discover()
    class_names = {c.name for c in classes}
    assert "hf_baseline" in class_names  # transformers is present
    for name in ("nnsight", "repeng", "pyvene"):
        assert name not in class_names
    out = capsys.readouterr().out
    for name, pkg in (("nnsight", "nnsight"), ("repeng", "repeng"), ("pyvene", "pyvene")):
        assert f"{name}: SKIPPED ({pkg} not installed)" in out


def test_discover_filter_by_name(monkeypatch) -> None:
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: True)
    classes = reg.discover(filter_names=["transformerlens"])
    assert [c.name for c in classes] == ["transformerlens"]


def test_discover_skips_missing_package(monkeypatch, capsys) -> None:
    # transformer_lens present, vllm missing.
    monkeypatch.setattr(
        reg, "is_package_available", lambda pkg: pkg != "vllm"
    )
    classes = reg.discover()
    class_names = {c.name for c in classes}
    assert "vllm" not in class_names
    assert "transformerlens" in class_names
    out = capsys.readouterr().out
    assert "vllm: SKIPPED" in out
    assert "vllm" in out


def test_discover_filters_by_required_capabilities(monkeypatch, capsys) -> None:
    monkeypatch.setattr(reg, "is_package_available", lambda pkg: True)
    # Only vllm advertises batching; transformerlens must be filtered out.
    classes = reg.discover(required=Capabilities(batching=True))
    class_names = {c.name for c in classes}
    assert "vllm" in class_names
    assert "transformerlens" not in class_names
    out = capsys.readouterr().out
    assert "transformerlens: SKIPPED" in out


def test_is_package_available_none_is_true() -> None:
    # An engine with no required package (required_package=None) is always ok.
    assert base_mod  # ensure import path is exercised
    assert reg.is_package_available(None) is True
