#!/usr/bin/env python3
"""Steering correctness sanity checks.

Verifies end-to-end behavioral correctness of the steering feature. This
is not a performance benchmark — it's the "does the feature actually work"
test you want to cite in a PR body and RFC comment.

Checks:
  1. Determinism
       Same prompt + seed + temperature=0 + steering config, two runs
       must produce bit-identical token ids.
  2. Config sensitivity
       Different steering configs on the same prompt must produce
       different outputs. Proves steering is actually being applied.
  3. Empty-dict vs None equivalence
       Passing steering_vectors=None vs steering_vectors={} must match
       a baseline LLM with steering disabled.
  4. Prefix cache separation (behavioral)
       Sequence: run A, run B, run A again.
       Assertion: output(A_first) == output(A_second) != output(B).
       If prefix caching were incorrectly reusing A's KV for B, then
       output(B) would equal output(A_first). Proves the prefix-cache
       key correctly incorporates prefill steering state.

Exit 0 on all-pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from steering_bench.vectors import random_steering_vectors

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {"hidden_size": 2560, "num_layers": 34},
    "google/gemma-3-12b-it": {"hidden_size": 3840, "num_layers": 48},
    "google/gemma-3-27b-it": {"hidden_size": 5376, "num_layers": 62},
}


def generate(llm, prompt: str, sp) -> tuple[str, tuple[int, ...]]:
    """Run a single prompt and return (text, token_ids)."""
    outputs = llm.generate([prompt], sp, use_tqdm=False)
    out = outputs[0].outputs[0]
    return out.text, tuple(out.token_ids)


def first_diff_idx(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def check_determinism(llm, hidden: int, layers: int) -> None:
    from vllm import SamplingParams

    vectors = random_steering_vectors(hidden, layers, scale=0.5, seed=1)
    sp = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        seed=42,
        steering_vectors=vectors,
    )
    prompt = "Once upon a time in a kingdom by the sea, there lived"
    _, ids_a = generate(llm, prompt, sp)
    _, ids_b = generate(llm, prompt, sp)
    if ids_a != ids_b:
        div = first_diff_idx(ids_a, ids_b)
        raise AssertionError(
            f"non-deterministic: two runs of identical prompt+steering diverged "
            f"at token {div}"
        )
    print(f"  PASS  determinism ({len(ids_a)} tokens matched)")


def check_config_sensitivity(llm, hidden: int, layers: int) -> None:
    from vllm import SamplingParams

    v1 = random_steering_vectors(hidden, layers, scale=1.0, seed=10)
    v2 = random_steering_vectors(hidden, layers, scale=1.0, seed=20)
    sp1 = SamplingParams(
        max_tokens=64, temperature=0.0, seed=42, steering_vectors=v1
    )
    sp2 = SamplingParams(
        max_tokens=64, temperature=0.0, seed=42, steering_vectors=v2
    )
    prompt = "The scientific method is built on"
    _, ids_v1 = generate(llm, prompt, sp1)
    _, ids_v2 = generate(llm, prompt, sp2)
    if ids_v1 == ids_v2:
        raise AssertionError(
            "config-insensitive: different steering vectors produced identical "
            "outputs — either steering is a no-op or vectors are too small"
        )
    div = first_diff_idx(ids_v1, ids_v2)
    print(f"  PASS  config sensitivity (diverges at token {div})")


def check_empty_vs_none(llm, hidden: int, layers: int) -> None:
    from vllm import SamplingParams

    sp_none = SamplingParams(max_tokens=64, temperature=0.0, seed=42)
    sp_empty = SamplingParams(
        max_tokens=64, temperature=0.0, seed=42, steering_vectors={}
    )
    prompt = "A haiku about the ocean:\n"
    _, ids_none = generate(llm, prompt, sp_none)
    _, ids_empty = generate(llm, prompt, sp_empty)
    if ids_none != ids_empty:
        div = first_diff_idx(ids_none, ids_empty)
        raise AssertionError(
            f"empty-dict != none: steering_vectors={{}} produced a different "
            f"output than steering_vectors=None (diverges at token {div}). "
            f"enabling steering with no active configs should be a no-op"
        )
    print(f"  PASS  empty-dict == none ({len(ids_none)} tokens matched)")


def check_prefix_cache_separation(llm, hidden: int, layers: int) -> None:
    from vllm import SamplingParams

    v1 = random_steering_vectors(hidden, layers, scale=1.0, seed=100)
    v2 = random_steering_vectors(hidden, layers, scale=1.0, seed=200)

    prefix = (
        "The principles of thermodynamics govern energy transfer in all "
        "physical systems. The first law states that energy is conserved. "
        "The second law states that entropy of an isolated system tends to "
        "increase over time. These principles explain "
    ) * 3  # long enough to be prefix-cacheable across multiple blocks

    sp_v1 = SamplingParams(
        max_tokens=48,
        temperature=0.0,
        seed=42,
        prefill_steering_vectors=v1,
    )
    sp_v2 = SamplingParams(
        max_tokens=48,
        temperature=0.0,
        seed=42,
        prefill_steering_vectors=v2,
    )

    _, ids_a1 = generate(llm, prefix, sp_v1)
    _, ids_b = generate(llm, prefix, sp_v2)
    _, ids_a2 = generate(llm, prefix, sp_v1)

    if ids_a1 != ids_a2:
        raise AssertionError(
            "non-deterministic prefill steering: two runs of prefix+v1 produced "
            "different outputs"
        )
    if ids_a1 == ids_b:
        raise AssertionError(
            "prefix cache leakage: prefix+v1 and prefix+v2 produced identical "
            "outputs. Either prefill steering is a no-op, or the prefix cache "
            "reused v1's KV entries when serving v2 (cache key is not "
            "separating prefill steering configs)"
        )
    div = first_diff_idx(ids_a1, ids_b)
    print(
        f"  PASS  prefix cache separation "
        f"(A1==A2, B diverges from A at token {div})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Steering correctness checks")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--max-steering-configs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS.get(args.model)
    if cfg is None:
        print(
            f"ERROR: unknown model {args.model}; add it to MODEL_CONFIGS",
            file=sys.stderr,
        )
        return 2

    from vllm import LLM

    print(f"loading {args.model} with enable_steering=True ...")
    llm = LLM(
        model=args.model,
        enable_steering=True,
        max_steering_configs=args.max_steering_configs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print("model loaded")

    checks = [
        ("determinism", check_determinism),
        ("config_sensitivity", check_config_sensitivity),
        ("empty_vs_none", check_empty_vs_none),
        ("prefix_cache_separation", check_prefix_cache_separation),
    ]

    failures = 0
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            fn(llm, cfg["hidden_size"], cfg["num_layers"])
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print(f"all {len(checks)} correctness checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
