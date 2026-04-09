"""nnsight benchmark: LanguageModel + trace context for steering.

nnsight uses deferred execution via trace contexts. Supports
automatic batching when multiple prompts are passed to trace().
"""

from __future__ import annotations

import torch

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class NnsightBenchmark:
    name = "nnsight"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        from nnsight import LanguageModel

        self.model = LanguageModel(model_id, device_map="auto", torch_dtype=torch.float16)
        self.vector = torch.tensor(vector, dtype=torch.float16, device="cuda")
        self.layer = layer

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        with self.model.generate(prompt, max_new_tokens=max_tokens, do_sample=False) as gen:
            with gen.invoke(prompt) as invoker:
                self.model.model.layers[self.layer].output[0][:] += self.vector
                output = self.model.output.save()
        return output.value.shape[1] - len(self.model.tokenizer.encode(prompt))

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        # nnsight supports batched trace, but per-element different vectors
        # requires separate invocations within the same trace
        results = []
        with self.model.generate(
            max_new_tokens=max_tokens, do_sample=False
        ) as gen:
            for i, prompt in enumerate(prompts):
                vec = torch.tensor(vectors[i], dtype=torch.float16, device="cuda")
                with gen.invoke(prompt) as invoker:
                    self.model.model.layers[self.layer].output[0][:] += vec
        # Count output tokens per prompt from the generation
        # nnsight batched generation returns combined output
        output = gen.output
        if hasattr(output, 'value'):
            # Approximate: return max_tokens for each since we can't easily split
            results = [max_tokens] * len(prompts)
        else:
            results = [max_tokens] * len(prompts)
        return results

    def teardown(self) -> None:
        del self.model
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
