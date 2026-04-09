# Design: How Each Benchmark Works

## vLLM Steering Internals (Context for Benchmarks)

The steering feature adds 3 operations per decoder layer per forward step:

```python
# In each decoder layer's forward():
residual = torch.ops.vllm.apply_steering(residual, self.steering_table_pre_attn, self.steering_index)
# ... attention ...
residual = torch.ops.vllm.apply_steering(residual, self.steering_table_post_attn, self.steering_index)
# ... MLP ...
residual = torch.ops.vllm.apply_steering(residual, self.steering_table_post_mlp, self.steering_index)
```

The custom op does: `hidden_states + steering_table[steering_index[:N]].to(hidden_states.dtype)`

This is an indexed gather (one row per token from a small table) followed by an add. The table has:
- Row 0: zeros (no-steering sentinel)
- Row 1: global prefill effective vector
- Row 2: global decode effective vector
- Rows 3+: per-request vectors (deduplicated by content hash)

Per-step Python overhead (before forward pass):
1. `SteeringManager.populate_steering_tables()` — copies vectors into table buffers across all layers x hook points
2. Index building loop — walks requests, maps each token to its table row
3. Phase transition handling — releases prefill configs, registers decode configs

The op is registered as opaque to `torch.compile` (prevents constant-folding zero buffers at trace time). CUDA graphs replay with live buffer values since buffers are updated in-place before each replay.

---

## Microbenchmarks

### `micro/steering_op.py`

**What it measures:** Raw latency of `torch.ops.vllm.apply_steering` in isolation.

**How:**
1. Import `torch.ops.vllm.apply_steering` (requires vLLM installed)
2. Create GPU tensors: `hidden_states(num_tokens, hidden_size)`, `steering_table(num_rows, hidden_size)`, `steering_index(num_tokens)`
3. Fill with random data, index values in `[0, num_rows)`
4. Warmup loop (100 iters) with `torch.cuda.synchronize()`
5. Timed loop (1000 iters): `synchronize(); start; op(); synchronize(); end`
6. Sweep: hidden_size x num_tokens x num_rows x dtype

**Why this matters:** Establishes the floor cost. If the op takes 2us per call and there are 3 calls per layer x 26 layers = 78 calls per step, that's 156us minimum overhead.

**Parameter matrix:**

| hidden_size | num_tokens | num_table_rows | dtype |
|-------------|-----------|----------------|-------|
| 2048 | 1, 32, 128, 512, 2048 | 4, 8, 16, 32 | fp16, bf16 |
| 3072 | 1, 32, 128, 512, 2048 | 4, 8, 16, 32 | fp16, bf16 |
| 4096 | 1, 32, 128, 512, 2048 | 4, 8, 16, 32 | fp16, bf16 |

### `micro/steering_manager.py`

**What it measures:** Python-side overhead of `SteeringManager` operations.

**How:**
1. Import `SteeringManager` from `vllm.v1.worker.steering_manager`
2. Create mock steerable layers (nn.Module with steering buffers)
3. Time individual methods: `register_config()`, `release_config()`, `populate_steering_tables()`, `get_row_for_config()`
4. Sweep: num_layers x num_configs x num_hook_points

**Why this matters:** `populate_steering_tables()` runs every step. It iterates `layers x hook_points` and does tensor copies. This is pure Python overhead that doesn't parallelize with GPU work.

### `micro/index_building.py`

**What it measures:** The inner loop of `_update_steering_state()` that fills `steering_index`.

**How:**
1. Create a `steering_index` tensor on GPU
2. Simulate the loop: for each request, write a row index for N tokens
3. Time with varying request counts and token counts
4. Include mixed-phase scenarios (some prefill, some decode)

**Why this matters:** This loop runs on CPU, touching a GPU tensor. If the loop itself is slow, it could become a bottleneck at high batch sizes.

---

## vLLM System Benchmarks

### `vllm_bench/latency.py`

**What it measures:** End-to-end per-request latency with and without steering.

**How:**
1. Instantiate `vllm.LLM` with `enable_steering=True/False`
2. Create `SamplingParams` with/without `steering_vectors`
3. Warmup: run 10 generate() calls
4. Measure: run 30 generate() calls, time each
5. Compare: steering_on latency vs steering_off latency = overhead

**Steering modes tested:**
- `none`: steering disabled entirely (`enable_steering=False`)
- `enabled_idle`: steering enabled, no vectors set (measures buffer allocation + zero-path overhead)
- `global_only`: steering enabled, global vectors set via `collective_rpc("set_steering_vectors")`
- `per_request_1`: one unique steering config via `SamplingParams.steering_vectors`
- `per_request_4`: four distinct configs in one batch

### `vllm_bench/throughput.py`

**What it measures:** Total tokens/sec when processing batches with diverse steering configs.

**How:**
1. Generate N prompts with M distinct steering configs
2. Submit all to `llm.generate()` in one call
3. Time total processing, compute tokens/sec
4. Vary N and M to understand deduplication benefit

### `vllm_bench/serving.py`

**What it measures:** Online serving metrics (TTFT, TPOT, ITL) with steering.

**How:**
1. Start `vllm serve` as subprocess with `--enable-steering`
2. Wait for health check
3. Send async HTTP requests via `aiohttp` to OpenAI-compatible endpoint
4. Include steering vectors in `extra_body.steering_vectors`
5. Measure standard serving metrics
6. Kill subprocess

**Why subprocess:** Matches real deployment. HTTP overhead is part of the measurement, and we want to test the full stack including the API router's steering validation.

### `vllm_bench/memory.py`

**What it measures:** GPU memory cost of steering buffers.

**How:**
1. `torch.cuda.reset_peak_memory_stats()`
2. Load model with `enable_steering=False`, record `torch.cuda.memory_allocated()`
3. Unload, clear cache
4. Load model with `enable_steering=True, max_steering_configs=N`, record memory
5. Delta = steering_on - steering_off
6. Validate against theoretical formula

**Theoretical formula:**
```
per_layer = 3 hooks * (max_configs + 3) * hidden_size * dtype_bytes   # tables
          + 3 hooks * 1 * hidden_size * dtype_bytes                     # vector buffers
shared    = max_batched_tokens * 8 bytes                                # steering_index (int64)
total     = num_layers * per_layer + shared
```

### `vllm_bench/startup.py`

**What it measures:** Model load time with/without steering buffer allocation.

**How:**
1. Run in subprocess for isolation (pattern from vLLM's own startup benchmark)
2. Time from `LLM()` constructor start to completion
3. Compare steering_on vs steering_off
4. Cold (no cache) and warm (cached weights) variants

---

## Ablation Benchmarks

Each ablation isolates one optimization variable and measures its interaction with steering.

### `ablation/cuda_graphs.py`

**Design:** 2x2 matrix — `(enforce_eager T/F) x (enable_steering T/F)`

Runs `vllm_bench/latency.py` for each of 4 configs. Reports:
- Baseline: no steering, CUDA graphs on
- Graph overhead without steering: eager vs graphs, no steering
- Steering overhead with graphs: graphs on, steering on vs off
- Steering overhead without graphs: eager, steering on vs off
- **Interaction effect:** does steering make CUDA graphs less effective?

The steering op is an opaque splitting point — it creates graph-break boundaries between compiled segments. This ablation measures whether that matters.

### `ablation/compile_level.py`

**Design:** 2x2 matrix — `(compilation_config.level 0/3) x (enable_steering T/F)`

Same structure. Tests whether torch.compile's Inductor optimizations interact with the opaque steering op.

### `ablation/prefix_cache.py`

**Design:** 2x3 matrix — `(enable_prefix_caching T/F) x (steering: none/global/per_request)`

Per-request prefill steering vectors are included in the cache key hash. This means different steering configs create different cache entries. This ablation measures:
- Cache hit rate degradation with diverse steering configs
- Whether decode-only steering (not in cache key) avoids this cost

### `ablation/config_scaling.py`

**Design:** 1xN sweep — `max_steering_configs: 1, 2, 4, 8, 16, 32`

With a fixed workload of K distinct configs per batch. Measures:
- Per-step overhead scaling (larger tables = more copies in populate)
- Memory scaling (linear with max_configs)
- Whether there's a cliff where overhead becomes significant

### `ablation/hook_points.py`

**Design:** 1x3 sweep — active hooks: 1 (post_mlp only), 2 (post_attn + post_mlp), 3 (all)

Set non-zero vectors only at selected hook points. Others remain zero.
Even zero-path calls execute the gather+add, so this measures whether the number of non-zero hooks matters (it shouldn't, since the op cost is dominated by memory access regardless of values).

### `ablation/model_family.py`

**Design:** 1xN sweep — same benchmark on different model architectures.

Runs the latency benchmark on Gemma-3-4B, LLaMA-3.2-1B, and Qwen3-0.6B (if small enough for 3090). Validates that overhead is consistent and not architecture-dependent.

---

## External Library Comparison

### Protocol

All external benchmarks implement a common interface:

```python
class SteeringBenchmark(Protocol):
    name: str

    def setup(self, model_id: str, vector: list[float], layer: int, hook: str) -> None:
        """Load model and configure steering."""
        ...

    def generate_single(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        """Generate with steering applied. Single request."""
        ...

    def generate_batch(self, prompts: list[list[int]], vectors: list[list[float]],
                       max_tokens: int) -> list[list[int]]:
        """Generate with per-prompt steering. Batch of N requests."""
        ...

    def teardown(self) -> None:
        """Unload model, free GPU memory."""
        ...

    def memory_allocated_mb(self) -> float:
        """Current GPU memory usage."""
        ...
```

### `external/hf_baseline.py`

**Purpose:** Baseline with no steering library. Pure `AutoModelForCausalLM.generate()`. All steering library overhead is relative to this.

### `external/transformerlens_bench.py`

**Setup:** `HookedTransformer.from_pretrained(model_id)`

**Single-request steering:**
```python
def hook_fn(activation, hook):
    activation[:, :, :] += steering_vector
    return activation

output = model.run_with_hooks(
    prompt_tokens,
    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
)
```

**Batching:** Sequential loop (TransformerLens doesn't support batch hooks well).

**Known overhead:** Hook registration/cleanup per call, Python callback dispatch per layer.

### `external/nnsight_bench.py`

**Setup:** `nnsight.LanguageModel(model_id)`

**Single-request steering:**
```python
with model.trace(prompt):
    model.model.layers[layer].output[0][:] += steering_vector
    output = model.output.save()
```

**Batching:** nnsight has automatic batching:
```python
with model.trace([prompt1, prompt2, ...]):
    model.model.layers[layer].output[0][:] += steering_vector
    output = model.output.save()
```

**Known overhead:** Trace context setup, deferred execution graph.

### `external/repeng_bench.py`

**Setup:** Load model via `transformers`, create `ControlVector` from steering vector.

**Single-request steering:**
```python
model.set_control(control_vector, coeff=1.0)
output = model.generate(input_ids, max_new_tokens=max_tokens)
```

**Batching:** Standard PyTorch batched generate (repeng patches the model's forward, so batching works naturally).

**Known overhead:** Very low — repeng does the same vector addition operation as vLLM's steering. This is the most direct comparison.

### `external/pyvene_bench.py`

**Setup:**
```python
config = pyvene.IntervenableConfig(
    representations=[pyvene.RepresentationConfig(
        layer=layer, component="block_output",
        intervention_type=pyvene.AdditionIntervention,
    )]
)
intervenable = pyvene.IntervenableModel(config, model)
```

**Single-request steering:**
```python
output = intervenable(
    base={"input_ids": input_ids},
    sources=[{"input_ids": input_ids}],
    intervention_args={"steering_vector": steering_vector},
)
```

**Batching:** Sequential (pyvene's intervention graph doesn't optimize batch dispatch).

### `external/vllm_single.py`

**Purpose:** vLLM in single-request mode for apples-to-apples comparison with the above.

```python
llm = vllm.LLM(model=model_id, enable_steering=True, max_steering_configs=4)
sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                    steering_vectors={"post_mlp": {layer: vector}})
output = llm.generate([prompt], [sp])
```

### `external/vllm_batched.py`

**Purpose:** Show what continuous batching buys you. N=16 requests with N distinct steering vectors.

```python
prompts = [prompt] * 16
params = [SamplingParams(steering_vectors={"post_mlp": {layer: vectors[i]}}) for i in range(16)]
outputs = llm.generate(prompts, params)
```

### Tier 1 vs Tier 2

**Tier 1 (single-request):** Every library processes one prompt with one steering vector. Measures: tokens/sec, ms/token, total latency, peak GPU memory. This is the fair apples-to-apples comparison.

**Tier 2 (batched, N=16):** Libraries that support batching use it. Those that don't fall back to sequential loops. Measures: total wall-clock for 16 requests, throughput (req/sec), average per-request latency. This shows the vLLM continuous-batching advantage.

---

## Analysis & Plotting

### `analysis/aggregate.py`

Reads all `results/**/*.json`, validates schema, merges into a pandas DataFrame with columns:
`benchmark, tag, model, parameter_*, result_*`

Computes derived columns:
- `overhead_pct`: `(steering_on - steering_off) / steering_off * 100`
- `speedup_vs_baseline`: `baseline_time / library_time`
- `memory_overhead_mb`: `steering_on_memory - steering_off_memory`

### `analysis/plot_overhead.py`

Grouped bar chart: batch_size on x-axis, latency on y-axis, bars grouped by steering mode (off, idle, global, per_request). One chart per model.

### `analysis/plot_comparison.py`

Two charts:
1. **Tier 1:** Grouped bars — one bar per library, y-axis = tokens/sec for single-request steering
2. **Tier 2:** Grouped bars — one bar per library, y-axis = total time for 16-request batch

### `analysis/plot_ablation.py`

Heatmap: rows = optimization on/off, columns = steering on/off, cells = latency. One heatmap per ablation (CUDA graphs, compile, prefix cache).

Tornado chart: bars showing the impact of each factor (CUDA graphs, compile, prefix cache, num hooks, num configs) sorted by magnitude.

### `analysis/plot_scaling.py`

Line plots:
1. x = max_steering_configs, y = latency (with error bands)
2. x = batch_size, y = throughput, lines for steering_off vs steering_on
3. x = num_active_hooks, y = per-step overhead

---

## Statistical Methodology

| Benchmark Group | Warmup Iters | Measured Iters | Why |
|----------------|-------------|---------------|-----|
| Micro | 100 | 1000 | Kernel timing is noisy, need many samples |
| System | 10 | 30 | Sufficient for CLT, each iter is expensive |
| External | 5 | 20 | Model loading is slow, each iter is expensive |

**All benchmarks report:** mean, median, p10, p25, p50, p75, p90, p99, stddev, N

**Outlier handling:** Flag (don't remove) samples > 3 stddev from mean. Report both with and without outliers.

**Environment control:**
- Pin GPU clocks: `nvidia-smi -lgc <base>,<base>` to eliminate boost variability
- Single GPU: `CUDA_VISIBLE_DEVICES=0`
- No competing workloads: verify via `nvidia-smi` before benchmark
- Record: GPU model, UUID, driver, CUDA, torch, vllm commit, python version

**Reproducibility:** Fixed random seed (42) for all vector generation. Full environment metadata in every JSON output file.
