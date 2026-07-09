"""Model-config resolution tests for the harness.

Run WITHOUT a GPU and WITHOUT transformers installed: the AutoConfig path is
exercised by injecting a fake ``transformers`` module into ``sys.modules``.
"""

from __future__ import annotations

import sys
import types

import pytest

from steering_bench.harness.models import (
    MODEL_CONFIGS,
    ModelConfig,
    ModelConfigError,
    get_model_config,
)

# -- static table --------------------------------------------------------------


def test_known_model_from_table() -> None:
    cfg = get_model_config("google/gemma-3-4b-it")
    assert cfg == ModelConfig(hidden_size=2560, num_layers=34)


@pytest.mark.parametrize(
    ("model", "hidden_size", "num_layers"),
    [
        ("Qwen/Qwen3-0.6B", 1024, 28),
        ("google/gemma-3-4b-it", 2560, 34),
        ("google/gemma-3-12b-it", 3840, 48),
        ("google/gemma-3-27b-it", 5376, 62),
        ("meta-llama/Llama-3.2-1B", 2048, 16),
        ("meta-llama/Llama-3.1-8B", 4096, 32),
    ],
)
def test_table_dims(model: str, hidden_size: int, num_layers: int) -> None:
    cfg = get_model_config(model)
    assert cfg.hidden_size == hidden_size
    assert cfg.num_layers == num_layers


def test_table_entries_are_frozen() -> None:
    cfg = MODEL_CONFIGS["meta-llama/Llama-3.2-1B"]
    with pytest.raises(Exception):
        cfg.hidden_size = 1  # type: ignore[misc]


# -- AutoConfig fallback -------------------------------------------------------


def _install_fake_transformers(monkeypatch, config_obj: object) -> list[str]:
    """Install a fake ``transformers`` module whose AutoConfig returns
    ``config_obj``. Returns a list that records the model_ids requested."""
    seen: list[str] = []

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **_: object) -> object:
            seen.append(model_id)
            return config_obj

    fake = types.ModuleType("transformers")
    fake.AutoConfig = _AutoConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return seen


def test_unknown_model_uses_autoconfig_standard_attrs(monkeypatch) -> None:
    stub = types.SimpleNamespace(hidden_size=1280, num_hidden_layers=20)
    seen = _install_fake_transformers(monkeypatch, stub)

    cfg = get_model_config("some/unknown-model")
    assert cfg == ModelConfig(hidden_size=1280, num_layers=20)
    assert seen == ["some/unknown-model"]


def test_unknown_model_uses_autoconfig_alt_attrs(monkeypatch) -> None:
    # GPT-2-style config: n_embd / n_layer instead of hidden_size /
    # num_hidden_layers.
    stub = types.SimpleNamespace(n_embd=768, n_layer=12)
    _install_fake_transformers(monkeypatch, stub)

    cfg = get_model_config("some/gpt2-style")
    assert cfg == ModelConfig(hidden_size=768, num_layers=12)


def test_autoconfig_missing_attrs_raises(monkeypatch) -> None:
    stub = types.SimpleNamespace(vocab_size=1000)  # no dims at all
    _install_fake_transformers(monkeypatch, stub)

    with pytest.raises(ModelConfigError) as exc:
        get_model_config("some/broken-model")
    msg = str(exc.value)
    assert "some/broken-model" in msg
    assert "hidden" in msg.lower() or "layer" in msg.lower()


def test_autoconfig_partial_attrs_raises(monkeypatch) -> None:
    # Has hidden size but no layer count.
    stub = types.SimpleNamespace(hidden_size=1024)
    _install_fake_transformers(monkeypatch, stub)

    with pytest.raises(ModelConfigError):
        get_model_config("some/half-model")
