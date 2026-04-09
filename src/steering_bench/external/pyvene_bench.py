"""pyvene benchmark: IntervenableModel with AdditionIntervention.

pyvene uses a config-driven intervention graph. Batching falls
back to sequential since the intervention graph doesn't optimize
batch dispatch.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class PyveneBenchmark:
    name = "pyvene"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        import pyvene

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

        # Map hook name to pyvene component
        component_map = {
            "pre_attn": "block_input",
            "post_attn": "block_output",
            "post_mlp": "block_output",
        }
        component = component_map.get(hook, "block_output")

        config = pyvene.IntervenableConfig(
            representations=[
                pyvene.RepresentationConfig(
                    layer=layer,
                    component=component,
                    intervention_type=pyvene.AdditionIntervention,
                )
            ]
        )
        self.intervenable = pyvene.IntervenableModel(config, model)
        self.vector = torch.tensor(vector, dtype=torch.float16, device="cuda")
        self.model = model

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # pyvene intervention for generation — apply at each forward step
        with torch.no_grad():
            # Use the intervenable model's generate method
            base = {"input_ids": inputs["input_ids"]}
            # Reshape vector for intervention: (batch, seq, hidden)
            intervention_vec = self.vector.unsqueeze(0).unsqueeze(0)
            outputs = self.intervenable.generate(
                base,
                [intervention_vec],
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # outputs is a tuple; first element contains generated tokens
        if isinstance(outputs, tuple):
            output_ids = outputs[0]
        else:
            output_ids = outputs
        return output_ids.shape[1] - input_len

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        # Sequential — pyvene doesn't optimize batch intervention dispatch
        results = []
        for i, prompt in enumerate(prompts):
            self.vector = torch.tensor(vectors[i], dtype=torch.float16, device="cuda")
            results.append(self.generate_single(prompt, max_tokens))
        return results

    def teardown(self) -> None:
        del self.intervenable
        del self.model
        del self.tokenizer
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
