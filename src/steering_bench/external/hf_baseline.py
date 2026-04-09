"""HuggingFace baseline: pure generate() with no steering.

This is the floor. All steering library overhead is relative to this.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class HFBaselineBenchmark:
    name = "hf_baseline"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs.shape[1] - inputs["input_ids"].shape[1]

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return [outputs.shape[1] - input_len] * len(prompts)

    def teardown(self) -> None:
        del self.model
        del self.tokenizer
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
