"""vLLM batched benchmark: continuous batching with per-request steering.

This showcases vLLM's key advantage: N distinct steering configs
processed in one generate() call via continuous batching.
"""

from __future__ import annotations

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class VllmBatchedBenchmark:
    name = "vllm_batched"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=model_id,
            enable_steering=True,
            max_steering_configs=16,  # Need room for diverse configs
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )
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
        from vllm import SamplingParams

        # True continuous batching: all requests in one generate() call
        sp_list = []
        for vec in vectors:
            vec_dict = {self.hook: {self.layer: vec}}
            sp = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                steering_vectors=vec_dict,
            )
            sp_list.append(sp)

        outputs = self.llm.generate(prompts, sp_list)
        return [len(o.outputs[0].token_ids) for o in outputs]

    def teardown(self) -> None:
        del self.llm
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
