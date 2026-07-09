"""Legacy external-library modules retained for the patch-sweep axis.

The cross-engine steering adapters that once lived here (hf_baseline, nnsight,
repeng, pyvene, transformerlens, vllm_single/batched) were ported to the
``SteeringEngine`` seam under ``steering_bench.engine.engines`` and retired.

What remains are the activation-patching / causal-tracing benchmarks -
``tl_patching`` and ``vllm_patch_sweep`` - which are a separate axis migrated
onto the seam in a later phase.
"""

from __future__ import annotations
