"""Single source of truth for model dimensions.

``get_model_config(model_id)`` resolves ``hidden_size`` / ``num_layers`` for a
model in two steps:

1. A static fallback table (``MODEL_CONFIGS``) merged from the per-script copies
   that used to live in ``bench_latency.py``, ``bench_external.py`` and
   ``capture_consumers/runner.py``. Known models resolve offline.
2. Otherwise, derive from HuggingFace ``transformers.AutoConfig`` — imported
   lazily so this module is importable without ``transformers`` installed.

Attribute names vary across configs (``hidden_size``/``n_embd``/``d_model``,
``num_hidden_layers``/``n_layer``/``num_layers``); both families are handled and
a clear ``ModelConfigError`` is raised if neither is present.
"""

from __future__ import annotations

from dataclasses import dataclass


class ModelConfigError(ValueError):
    """Raised when a model's dimensions cannot be resolved."""


@dataclass(frozen=True)
class ModelConfig:
    """A model's steering-relevant dimensions."""

    hidden_size: int
    num_layers: int


# Static fallback table. Merged from the duplicated MODEL_CONFIGS copies.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Qwen/Qwen3-0.6B": ModelConfig(hidden_size=1024, num_layers=28),
    "google/gemma-3-4b-it": ModelConfig(hidden_size=2560, num_layers=34),
    "google/gemma-3-12b-it": ModelConfig(hidden_size=3840, num_layers=48),
    "google/gemma-3-27b-it": ModelConfig(hidden_size=5376, num_layers=62),
    "meta-llama/Llama-3.2-1B": ModelConfig(hidden_size=2048, num_layers=16),
    "meta-llama/Llama-3.1-8B": ModelConfig(hidden_size=4096, num_layers=32),
    "facebook/opt-125m": ModelConfig(hidden_size=768, num_layers=12),
    "facebook/opt-350m": ModelConfig(hidden_size=1024, num_layers=24),
}

# Attribute-name families, in priority order, for the AutoConfig fallback.
_HIDDEN_SIZE_ATTRS = ("hidden_size", "n_embd", "d_model", "hidden_dim")
_NUM_LAYER_ATTRS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")


def _first_attr(config: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _from_autoconfig(model_id: str) -> ModelConfig:
    """Derive a ``ModelConfig`` from ``transformers.AutoConfig`` (lazy import)."""
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ModelConfigError(
            f"model {model_id!r} is not in the static table and transformers is "
            "not installed to derive its config; install transformers or add it "
            "to steering_bench.harness.models.MODEL_CONFIGS"
        ) from exc

    config = transformers.AutoConfig.from_pretrained(model_id)

    hidden_size = _first_attr(config, _HIDDEN_SIZE_ATTRS)
    num_layers = _first_attr(config, _NUM_LAYER_ATTRS)
    missing: list[str] = []
    if hidden_size is None:
        missing.append(f"hidden size (tried {', '.join(_HIDDEN_SIZE_ATTRS)})")
    if num_layers is None:
        missing.append(f"layer count (tried {', '.join(_NUM_LAYER_ATTRS)})")
    if missing:
        raise ModelConfigError(
            f"cannot resolve {' and '.join(missing)} for model {model_id!r} from "
            f"its AutoConfig ({type(config).__name__})"
        )

    return ModelConfig(hidden_size=hidden_size, num_layers=num_layers)


def get_model_config(model_id: str) -> ModelConfig:
    """Resolve a model's ``hidden_size`` / ``num_layers``.

    Known models resolve offline from ``MODEL_CONFIGS``; any other model resolves
    via ``transformers.AutoConfig`` if it can be fetched. Raises
    ``ModelConfigError`` if dimensions cannot be determined.
    """
    known = MODEL_CONFIGS.get(model_id)
    if known is not None:
        return known
    return _from_autoconfig(model_id)
