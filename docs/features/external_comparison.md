# External Library Comparison

## Scope

Compare vLLM steering against TransformerLens, nnsight, repeng, and pyvene.

### In Scope
- Tier 1: single-request latency comparison across all libraries
- Tier 2: batched (N=16) throughput comparison
- HuggingFace baseline (no steering) as the floor
- vLLM single-request and batched modes

### Not In Scope
- Semantic quality of steering (only performance)
- Libraries not listed above

## Data/Control Flow

```
scripts/bench_external.py (runner)
  → Discovers installed libraries via importlib
  → For each library:
      bench = LibraryBenchmark()
      bench.setup(model_id, vector, layer, hook)
      
      Tier 1 (single-request):
        Warmup 5 iters
        Measure 20 iters of bench.generate_single(prompt, max_tokens)
        Timing done externally (time.perf_counter + cuda.synchronize)
        
      Tier 2 (batched N=16):
        Warmup 5 iters
        Measure 20 iters of bench.generate_batch(prompts, vectors, max_tokens)
        
      bench.teardown()
      gc.collect() + cuda.empty_cache()
      
      write_result("external.tier1.{name}", ...) 
      write_result("external.tier2.{name}", ...)
```

## Protocol

All implementations in `src/steering_bench/external/` implement:

```python
class SteeringBenchmark(Protocol):
    name: str
    def setup(model_id, vector, layer, hook) -> None
    def generate_single(prompt: str, max_tokens: int) -> int  # output token count
    def generate_batch(prompts, vectors, max_tokens) -> list[int]
    def teardown() -> None
    def memory_allocated_mb() -> float
```

Strings (not token IDs) are passed since each library tokenizes differently.

## Library Implementations

| Library | Batching | Steering Mechanism |
|---------|----------|-------------------|
| hf_baseline | HF batched generate | None (floor) |
| transformerlens | Sequential loop | Hook callbacks via run_with_hooks |
| nnsight | Automatic (trace context) | Deferred execution via trace |
| repeng | Sequential (per-vector) | Forward pass patching via ControlModel |
| pyvene | Sequential loop | Intervention graph via IntervenableModel |
| vllm_single | Sequential (apples-to-apples) | SamplingParams.steering_vectors |
| vllm_batched | Continuous batching | SamplingParams.steering_vectors per request |

## Files

| File | Purpose |
|------|---------|
| `src/steering_bench/external/base.py` | SteeringBenchmark Protocol + helpers |
| `src/steering_bench/external/hf_baseline.py` | HuggingFace baseline |
| `src/steering_bench/external/transformerlens_bench.py` | TransformerLens |
| `src/steering_bench/external/nnsight_bench.py` | nnsight |
| `src/steering_bench/external/repeng_bench.py` | repeng |
| `src/steering_bench/external/pyvene_bench.py` | pyvene |
| `src/steering_bench/external/vllm_single.py` | vLLM single-request |
| `src/steering_bench/external/vllm_batched.py` | vLLM continuous batching |
| `scripts/bench_external.py` | Runner script |

## Invariants

- Default model: meta-llama/Llama-3.2-1B (all external libs support LLaMA)
- 5 warmup / 20 measured iterations
- Libraries not installed are skipped gracefully
- Full GPU cleanup between libraries
- Timing done externally to the benchmark class for consistency
- `--libraries` flag allows filtering to specific libraries
