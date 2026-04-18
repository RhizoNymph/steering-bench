"""LLM factory and model-config helpers for capture consumer benchmarks."""

from __future__ import annotations

from typing import Any


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "facebook/opt-125m": {"hidden_size": 768, "num_layers": 12},
    "facebook/opt-350m": {"hidden_size": 1024, "num_layers": 24},
    "meta-llama/Llama-3.2-1B": {"hidden_size": 2048, "num_layers": 16},
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
}


def get_model_config(model: str) -> dict[str, int]:
    cfg = MODEL_CONFIGS.get(model)
    if cfg is None:
        print(f"Warning: unknown model {model!r}, defaulting to facebook/opt-125m config")
        return MODEL_CONFIGS["facebook/opt-125m"]
    return cfg


def make_llm(
    model: str,
    capture_consumers: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    from vllm import LLM
    return LLM(model=model, capture_consumers=capture_consumers, **kwargs)


def make_prompts(num_prompts: int, prompt_len: int) -> list[str]:
    words_needed = max(1, int(prompt_len / 1.3))
    base = " ".join(["hello"] * words_needed)
    return [base] * num_prompts
