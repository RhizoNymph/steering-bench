"""TransformerLens benchmark: HookedTransformer + run_with_hooks.

Steering is applied via Python hook callbacks registered per forward pass.
Batching falls back to sequential loop.
"""

from __future__ import annotations

import torch

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class TransformerLensBenchmark:
    name = "transformerlens"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        from transformer_lens import HookedTransformer

        self.model = HookedTransformer.from_pretrained(
            model_id, dtype="float16", device="cuda"
        )
        self.vector = torch.tensor(vector, dtype=torch.float16, device="cuda")
        self.layer = layer
        # TransformerLens hook naming
        hook_map = {
            "pre_attn": f"blocks.{layer}.hook_resid_pre",
            "post_attn": f"blocks.{layer}.hook_attn_out",
            "post_mlp": f"blocks.{layer}.hook_resid_post",
        }
        self.hook_name = hook_map.get(hook, f"blocks.{layer}.hook_resid_post")

    def _make_hook_fn(self, vector: torch.Tensor):
        def hook_fn(activation, hook):
            activation[:, :, :] += vector
            return activation
        return hook_fn

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        hook_fn = self._make_hook_fn(self.vector)
        tokens = self.model.to_tokens(prompt)
        input_len = tokens.shape[1]
        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                do_sample=False,
                fwd_hooks=[(self.hook_name, hook_fn)],
            )
        return output.shape[1] - input_len

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        # Sequential — TransformerLens doesn't support per-element hooks in batches
        results = []
        for i, prompt in enumerate(prompts):
            vec = torch.tensor(vectors[i], dtype=torch.float16, device="cuda")
            hook_fn = self._make_hook_fn(vec)
            tokens = self.model.to_tokens(prompt)
            input_len = tokens.shape[1]
            with torch.no_grad():
                output = self.model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    fwd_hooks=[(self.hook_name, hook_fn)],
                )
            results.append(output.shape[1] - input_len)
        return results

    def teardown(self) -> None:
        del self.model
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
