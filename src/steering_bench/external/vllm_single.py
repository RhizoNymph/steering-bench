"""vLLM single-request benchmark for apples-to-apples comparison.

Uses vLLM's LLM API with one request at a time to match the
evaluation conditions of external libraries.
"""

from __future__ import annotations

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class VllmSingleBenchmark:
    name = "vllm_single"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=model_id,
            enable_steering=True,
            max_steering_configs=4,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )
        # Convert flat vector to vLLM's nested format
        self.vectors_dict = {hook: {layer: vector}}
        self.layer = layer
        self.hook = hook

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        from vllm import SamplingParams

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            steering_vectors=self.vectors_dict,
        )
        outputs = self.llm.generate([prompt], [sp])
        return len(outputs[0].outputs[0].token_ids)

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        # Sequential single-request calls for apples-to-apples
        results = []
        for i, prompt in enumerate(prompts):
            from vllm import SamplingParams

            vec_dict = {self.hook: {self.layer: vectors[i]}}
            sp = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                steering_vectors=vec_dict,
            )
            outputs = self.llm.generate([prompt], [sp])
            results.append(len(outputs[0].outputs[0].token_ids))
        return results

    def teardown(self) -> None:
        del self.llm
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
