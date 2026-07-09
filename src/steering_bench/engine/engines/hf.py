"""HuggingFace baseline engine adapter.

Ports ``external/hf_baseline.py``: plain ``transformers`` greedy generation with
**no steering applied**.  It is the latency floor -- every steering library's
overhead is measured relative to it -- so a request's ``steering`` field is
accepted and deliberately ignored.

Real batching: prompts are padded and generated in one ``model.generate`` call.
Token counts are exact (``output_tokens_exact=True``).  ``transformers`` and
``torch`` are imported lazily inside methods so this module is importable with
neither installed.
"""

from __future__ import annotations

from typing import Any

from steering_bench.engine.base import Capabilities, SteeringConfig, SteeringEngine
from steering_bench.engine.spec import GenerationRequest, GenerationResult


class HFSteeringEngine(SteeringEngine):
    """Adapter over ``transformers.AutoModelForCausalLM`` -- the no-op floor."""

    name = "hf_baseline"
    capabilities = Capabilities(
        batching=True,
        named_modules=False,
        multi_layer=False,
        multi_hook=False,
        capture=False,
    )

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(
        self,
        model_id: str,
        *,
        steering_config: SteeringConfig | None = None,
        **opts: object,
    ) -> None:
        # No fork-specific steering knobs; steering_config is ignored (no-op).
        del steering_config
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = opts.pop("torch_dtype", torch.float16)
        device_map = opts.pop("device_map", "auto")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map, **opts
        )
        self._model.eval()

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HFSteeringEngine.generate called before load()")

        # No-op baseline: steering is accepted but ignored on purpose.
        prompts = [req.prompt for req in requests]
        max_tokens = max(req.max_tokens for req in requests)

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = outputs.shape[1] - input_len
        return [GenerationResult(output_tokens=new_tokens) for _ in requests]

    def memory_allocated_mb(self) -> float:
        return self._gpu_memory_mb()

    def teardown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cleanup_gpu()

    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("transformers")
        except PackageNotFoundError:
            return "unknown"
