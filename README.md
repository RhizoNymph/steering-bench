# steering-bench

A standalone benchmark harness that measures the performance overhead of vLLM's
activation steering feature. It compares vLLM steering against external libraries
(TransformerLens, nnsight, repeng, pyvene) and measures interactions with vLLM
optimizations (CUDA graphs, torch.compile, prefix caching).

The harness is fully external to vLLM — vLLM is an optional dependency.

## Install

Requires Python >= 3.10 and a CUDA-capable GPU. Uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync                          # core deps only
uv sync --extra vllm             # + vLLM (pulled from local editable path in pyproject)
uv sync --extra all-external     # + TransformerLens, nnsight, pyvene
uv sync --extra dev               # + pytest, ruff, mypy
```

`repeng` pins `numpy<2` which conflicts with vLLM, so install it in a separate
environment if needed:

```bash
uv pip install repeng
```

## Running benchmarks

Each script under `scripts/` is independently executable, writes a timestamped
JSON result into `results/<category>/`, and prints a human-readable summary to
stdout. Every script accepts `--help` for the authoritative list.

```bash
uv run scripts/bench_latency.py --model google/gemma-3-4b-it --batch-sizes 1,4,8,16
uv run scripts/bench_latency.py --help
```

H100 batch runner: `scripts/run_h100.sh`. Nsight profiling helpers:
`profile_cuda.sh`, `scripts/nsys_target.py`, `scripts/profile_steering.py`.

### Common arguments

Most benchmark scripts share the following flags. Defaults differ per script
(for example micro-benchmarks use `warmup=50–100`, `iters=500–1000`; vLLM
system benchmarks use `warmup=3–5`, `iters=5–20`).

| Flag | Purpose |
|---|---|
| `--model STR` | HF model id. Defaults to `google/gemma-3-4b-it` (external comparison defaults to `meta-llama/Llama-3.2-1B`). |
| `--output-dir PATH` | Where to write the JSON result. Defaults vary: `results/micro/`, `results/vllm/`, `results/ablation/`, `results/external/`, `results/serving/`, `results/profile/`. |
| `--warmup INT` | Unmeasured warmup iterations. |
| `--iters INT` | Measured iterations used for stats. |
| `--max-tokens INT` | Tokens to generate per request (default 128 for most, 64 for profile scripts). |
| `--batch-size INT` / `--batch-sizes LIST` | Single size or comma-separated sweep. |
| `--prompt-len INT` | Approximate prompt length in tokens (default 64). |
| `--gpu-memory-utilization FLOAT` | Upper bound on vLLM GPU memory. Defaults 0.6–0.9 depending on benchmark. |
| `--tag STR` | Suffix appended to the result filename for grouping runs. |

### Micro-benchmarks (`results/micro/`)

Raw primitives, no vLLM engine. All three accept `--warmup`, `--iters`,
`--device` (default `cuda:0`), `--output-dir`, `--tag`.

| Script | Purpose | Extra flags |
|---|---|---|
| `bench_steering_op.py` | Raw steering op kernel latency. | `--subset` (reduced sweep). Defaults: warmup 100, iters 1000. |
| `bench_steering_manager.py` | Python-side `SteeringManager` overhead. | Defaults: warmup 50, iters 500. |
| `bench_index_building.py` | `steering_index` construction loop. | Defaults: warmup 50, iters 500. |

### vLLM system benchmarks (`results/vllm/`)

End-to-end `LLM.generate()` timings.

| Script | Purpose | Distinctive flags |
|---|---|---|
| `bench_latency.py` | Per-request latency across steering modes × batch sizes. The headline overhead number. | `--batch-sizes 1,4,8,16`, `--disable-prefix-cache` |
| `bench_throughput.py` | Batch throughput as `distinct_configs` grows. | `--num-prompts 64`, `--configs-sweep 0,1,4,8`, `--max-steering-configs`, `--disable-prefix-cache` |
| `bench_throughput_matrix.py` | Mode × batch-size throughput matrix. | `--batch-sizes 1,4,8,16,32`, `--fractions 0.0,0.25,0.5,0.75,1.0`, `--max-steering-configs 4` |
| `bench_memory.py` | GPU memory cost of steering buffers. | `--configs-sweep 0,4,8,16,32`, `--num-gpu-blocks 64`, `--gpu-memory-utilization 0.6` |
| `bench_max_tokens.py` | Per-step overhead convergence across generation length. | `--max-tokens-list 64,128,256,512,1024,2048`, `--num-active-list 0,1,8,16`, `--max-steering-configs 32`, `--distinct-vectors` |
| `bench_mixed_batch.py` | Do non-steered requests in a mixed batch pay the steering cost? | `--distinct-vectors`, `--max-steering-configs`, `--num-active-only`, `--max-num-seqs` (needed for batch > 256) |
| `bench_table_sizing.py` | `max_steering_configs` × `batch_size` × `distinct_configs` sweep. | `--max-configs-sweep 4,16`, `--distinct-sweep 1,4,8`, `--batch-sizes 8,16,32`, `--skip-disabled`, `--skip-steering`. Default `--tag table-sizing`. |

### Ablations (`results/ablation/`)

Interactions with other vLLM optimizations.

| Script | Purpose | Distinctive flags |
|---|---|---|
| `bench_cuda_graphs.py` | Does CUDA graph capture still work with steering? | `--batch-sizes 1,4,8` |
| `bench_hook_points.py` | Scaling vs. number of active hook points. | `--batch-sizes 1,4,8` |
| `bench_config_scaling.py` | Scaling vs. `max_steering_configs`. | `--configs-sweep 1,2,4,8,16,32`, `--batch-size 8` |

### Online serving (`results/serving/`)

`bench_serving.py` spawns a `vllm serve` subprocess and measures TTFT / TPOT /
ITL / E2EL. Flags: `--python-bin .venv/bin/python`, `--port 8765`,
`--num-prompts 64`, `--concurrency 16`, `--max-tokens 256`, `--prompt-len 256`,
`--max-model-len 4096`, `--max-steering-configs 16`, `--startup-timeout 240`,
`--sharegpt-path PATH` (optional real prompts), and `--modes` (default
`disabled,enabled_idle,all_steered_shared,per_request_n4,per_request_n16`).

### External comparison (`results/external/`)

`bench_external.py` compares vLLM against TransformerLens, nnsight, repeng, and
pyvene. Default model `meta-llama/Llama-3.2-1B`. Flags: `--layer 8`,
`--hook post_mlp`, `--libraries all` (or comma-separated subset), `--skip-tier1`
(single-request), `--skip-tier2` (batched), plus the common timing/output flags.
Batch size for Tier 2 defaults to 16.

### Profiling and correctness

| Script | Purpose | Flags |
|---|---|---|
| `nsys_target.py` | Minimal generate() wrapper to attach `nsys profile` to. | `--mode {disabled,steering}` (required), `--batch-size 8`, `--num-active -1` (−1 = all), `--shared-vector`, `--max-tokens 64`, `--warmup 1`, `--iters 2`, `--gpu-memory-utilization 0.6` |
| `profile_steering.py` | Torch profiler trace of a full forward pass. Run twice, once per mode. | `--mode {disabled,steering}` (required), `--batch-size 8`, `--warmup 2`, `--iters 3` |
| `verify_correctness.py` | Steering sanity checks (output differs when enabled, etc.). | `--max-steering-configs 4`, `--max-model-len 4096`, `--gpu-memory-utilization 0.8` |

### Analysis and maintenance

| Script | Purpose | Flags |
|---|---|---|
| `analyze.py` | Aggregate results and generate matplotlib plots. | `--results-dir results/`, `--output-dir results/plots/`, `--format {png,svg,pdf}`, `--tag STR` (filter by tag) |
| `compare_throughput.py` | Compare throughput runs across tags. | `--results-dir results/vllm/`, `--tags baseline-v2,no-prefix-cache,big-table` |
| `migrate_throughput_keys.py` | One-time rename of `*_ms` → `*_tps` throughput stat keys. | `--results-dir results/`, `--dry-run` |

## Layout

```
src/steering_bench/     shared core: timing, output schema, vector generation
scripts/                runnable benchmarks and analysis
results/                JSON outputs (gitignored)
docs/                   design docs, roadmap, per-feature docs
```

See `docs/OVERVIEW.md` for the subsystem map and the feature index. Per-feature
detail lives in `docs/features/`:

- `core.md` — timing, result schema, vector generation
- `micro_benchmarks.md` — steering op, manager, index building
- `vllm_benchmarks.md` — latency, throughput, memory
- `ablation_benchmarks.md` — CUDA graphs, config scaling, hook points
- `external_comparison.md` — TransformerLens / nnsight / repeng / pyvene vs vLLM
- `analysis.md` — aggregation and plotting

## Result format

All benchmarks share a single JSON schema written by
`steering_bench.output.write_result`: environment metadata (GPU, CUDA, torch,
vllm versions), benchmark parameters, and timing statistics (mean, median,
p10/p50/p90/p99, stddev, raw samples). Timing uses GPU-synchronized CUDA events
where relevant.
