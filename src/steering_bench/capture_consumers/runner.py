"""LLM factory and model-config helpers for capture consumer benchmarks."""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from collections.abc import Callable
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


def make_prompts(
    num_prompts: int,
    prompt_len: int,
    model: str | None = None,
) -> list[str]:
    """Generate ``num_prompts`` prompts of approximately ``prompt_len`` tokens.

    When ``model`` is provided the prompt is tokenizer-exact: the model's
    tokenizer is used to produce a string whose encoded length is exactly
    ``prompt_len`` tokens (excluding BOS).  Without ``model`` a rough
    words-based heuristic is used (1 word ≈ 1.3 tokens for English).
    """
    if model is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        raw = "hello world " * (prompt_len * 4)
        ids = tok.encode(raw, add_special_tokens=False)[:prompt_len]
        prompt = tok.decode(ids)
        return [prompt] * num_prompts
    words_needed = max(1, int(prompt_len / 1.3))
    return [" ".join(["hello"] * words_needed)] * num_prompts


def _subprocess_entry(
    target: Callable[..., dict],
    kwargs: dict[str, Any],
    result_q: Any,
) -> None:
    """Child entry point: run ``target(**kwargs)`` and ship the dict back.

    Any exception is converted to the same ``{"error": ...}`` shape the
    benchmark cell functions return on failure, so the parent gets a uniform
    result regardless of whether the cell raised or returned an error dict.
    """
    try:
        result = target(**kwargs)
    except BaseException as exc:  # noqa: BLE001 — must not crash the child silently
        result = {"error": f"{type(exc).__name__}: {exc}", "samples_ms": []}
    result_q.put(result)


def run_in_subprocess(
    target: Callable[..., dict],
    kwargs: dict[str, Any],
    timeout_s: float = 900.0,
) -> dict:
    """Run ``target(**kwargs)`` in a fresh spawned process and return its dict.

    Each benchmark cell constructs and tears down a vLLM ``LLM``; running it in
    an isolated child means the engine's residual-weight allocations and CUDA
    context do not compound across configs (the parent would eventually OOM).
    A *spawn* context is mandatory: the parent has already initialised CUDA, and
    a forked child cannot re-init it.

    ``target`` and every value in ``kwargs`` must be picklable (spawn pickles
    them into the child). The child's return value is read off the queue before
    ``join`` to avoid the large-payload feeder-thread deadlock. On timeout the
    child is terminated and an ``{"error": ...}`` dict is returned instead of a
    benchmark result, matching the cell functions' failure convention.
    """
    ctx = mp.get_context("spawn")
    result_q: Any = ctx.Queue()
    proc = ctx.Process(
        target=_subprocess_entry,
        args=(target, kwargs, result_q),
    )
    proc.start()
    try:
        result: dict = result_q.get(timeout=timeout_s)
    except _queue.Empty:
        result = {
            "error": f"subprocess timeout after {timeout_s:.0f}s",
            "samples_ms": [],
        }
    finally:
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.terminate()
            proc.join()
    return result
