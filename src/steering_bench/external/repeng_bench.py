"""repeng benchmark: ControlModel + set_control.

repeng patches the model's forward pass to add steering vectors.
Architecturally closest to vLLM's approach — both do vector addition.
This is the most direct single-request comparison.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_bench.external.base import cleanup_gpu, gpu_memory_mb


class RepengBenchmark:
    name = "repeng"

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        from repeng import ControlModel, ControlVector

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

        # Wrap with ControlModel — specify which layers to control
        num_layers = model.config.num_hidden_layers
        # Control the target layer and a few around it
        control_layers = list(range(max(0, layer - 2), min(num_layers, layer + 3)))
        self.model = ControlModel(model, control_layers)

        # Create a ControlVector — repeng expects a dict of {layer: vector}
        vec_tensor = torch.tensor(vector, dtype=torch.float16)
        self.control_vector = ControlVector(
            model_type=model.config.model_type,
            directions={layer: vec_tensor},
        )
        self.model.set_control(self.control_vector, coeff=1.0)

    def generate_single(self, prompt: str, max_tokens: int) -> int:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs.shape[1] - input_len

    def generate_batch(
        self, prompts: list[str], vectors: list[list[float]], max_tokens: int
    ) -> list[int]:
        # repeng applies the same control vector to all batch elements.
        # For per-prompt different vectors, we must go sequential.
        results = []
        for i, prompt in enumerate(prompts):
            # Swap control vector for this prompt
            vec_tensor = torch.tensor(vectors[i], dtype=torch.float16)
            layer = list(self.control_vector.directions.keys())[0]
            cv = type(self.control_vector)(
                model_type=self.control_vector.model_type,
                directions={layer: vec_tensor},
            )
            self.model.set_control(cv, coeff=1.0)
            results.append(self.generate_single(prompt, max_tokens))
        return results

    def teardown(self) -> None:
        self.model.reset()
        del self.model
        del self.tokenizer
        cleanup_gpu()

    def memory_allocated_mb(self) -> float:
        return gpu_memory_mb()
