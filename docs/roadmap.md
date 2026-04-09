# Execution Roadmap

## Phase 0: Setup

**Prerequisites (manual, one-time per machine):**

```bash
# Create project, install deps
cd steering-bench
pip install -e ".[all]"            # or selective: ".[vllm]", ".[all-external]"

# Point at local vLLM fork if not on PyPI
pip install -e /path/to/vllm-fork && pip install -e "."

# Verify environment
steering-bench env

# Download models
huggingface-cli download meta-llama/Llama-3.2-1B
huggingface-cli download google/gemma-3-4b-it

# Pin GPU clocks for reproducibility
sudo nvidia-smi -lgc 1400,1400    # adjust for your GPU's base clock
```

**Estimated time:** 30 min (mostly model downloads)

---

## Phase 1: Microbenchmarks

**What:** Time the raw steering primitives without loading a model.
**Depends on:** vLLM installed (imports internals)
**Machine:** Any

```bash
steering-bench run micro --output-dir results/micro/
```

| Benchmark | What It Measures | Est. Time |
|-----------|-----------------|-----------|
| `steering_op` | `torch.ops.vllm.apply_steering` kernel latency across hidden_size x tokens x table_rows x dtype | 5 min |
| `steering_manager` | `SteeringManager` method timing (register, release, populate, get_row) | 3 min |
| `index_building` | steering_index construction loop across batch sizes | 2 min |

**Total: ~10 min**

**What we learn:** The absolute floor cost of steering. If the op takes Xus and there are 3 hooks x L layers = 3L calls/step, minimum per-step GPU overhead is 3L * Xus.

---

## Phase 2: vLLM System Benchmarks

**What:** End-to-end overhead of steering in real vLLM inference.
**Depends on:** vLLM + models downloaded
**Machine:** Gemma-3-4B on 3090

```bash
steering-bench run vllm --model google/gemma-3-4b-it --output-dir results/vllm/
```

| Benchmark | What It Measures | Est. Time |
|-----------|-----------------|-----------|
| `latency` | Per-request latency: steering off/idle/global/per_request, across batch sizes and sequence lengths | 60 min |
| `throughput` | Batch throughput with 0/1/4/8 distinct steering configs | 20 min |
| `serving` | Online serving TTFT/TPOT/ITL/E2EL with steering via HTTP | 30 min |
| `memory` | GPU memory delta for max_configs 0/4/8/16/32 | 10 min |
| `startup` | Cold/warm startup time with/without steering | 15 min |

**Total: ~2 hours**

**What we learn:** Real-world overhead percentages. "Steering adds X% latency" and "steering buffers cost Y MB."

---

## Phase 3: Ablation Matrix

**What:** Isolate how CUDA graphs, torch.compile, prefix caching, and other optimizations interact with steering.
**Depends on:** vLLM + models
**Machine:** Can share with Phase 2 (run sequentially) or separate machine

```bash
steering-bench run ablation --model google/gemma-3-4b-it --output-dir results/ablation/
```

| Benchmark | Design | Est. Time |
|-----------|--------|-----------|
| `cuda_graphs` | 2x2: enforce_eager x enable_steering | 20 min |
| `compile_level` | 2x2: compile level 0/3 x steering | 20 min |
| `prefix_cache` | 2x3: prefix_caching x steering mode | 20 min |
| `config_scaling` | 1x6: max_steering_configs 1,2,4,8,16,32 | 25 min |
| `hook_points` | 1x3: 1/2/3 active hooks per layer | 15 min |
| `model_family` | 1x3: Gemma3, LLaMA, Qwen (needs extra models) | 40 min |

**Total: ~2.5 hours**

**What we learn:** Which optimizations matter for steering, and whether steering degrades any existing optimization. The key question: "Does steering break CUDA graphs?"

---

## Phase 4: External Library Comparison

**What:** Compare vLLM steering against TransformerLens, nnsight, repeng, pyvene.
**Depends on:** External libs installed, LLaMA-3.2-1B downloaded
**Machine:** Independent of Phases 2-3

```bash
steering-bench run external --model meta-llama/Llama-3.2-1B --output-dir results/external/
```

| Benchmark | Tier 1 (single) | Tier 2 (batch N=16) | Est. Time |
|-----------|-----------------|---------------------|-----------|
| `hf_baseline` | HF generate, no steering | N/A (baseline) | 10 min |
| `transformerlens` | HookedTransformer + hooks | Sequential loop | 15 min |
| `nnsight` | trace + activation add | Automatic batching | 15 min |
| `repeng` | ControlVector + generate | Batched generate | 15 min |
| `pyvene` | IntervenableModel | Sequential loop | 15 min |
| `vllm_single` | vLLM single request | N/A | 10 min |
| `vllm_batched` | N/A | Continuous batching | 10 min |

**Total: ~1.5 hours**

**What we learn:** Where vLLM sits in the landscape. Single-request: how does steering overhead compare to hook-based approaches? Batched: how much does continuous batching help?

---

## Phase 5: Analysis

**What:** Aggregate results, generate plots and report.
**Depends on:** All previous phases complete

```bash
steering-bench analyze --results-dir results/ --output-dir results/reports/
steering-bench plot --results-dir results/ --output-dir results/plots/
```

| Output | Description |
|--------|-------------|
| `reports/summary.md` | Auto-generated report answering the 10 key questions |
| `reports/summary.csv` | Full results table |
| `plots/overhead_*.png` | Steering on/off bar charts by batch size and model |
| `plots/comparison_tier1.png` | Cross-library single-request comparison |
| `plots/comparison_tier2.png` | Cross-library batched comparison |
| `plots/ablation_heatmap.png` | Optimization interaction heatmap |
| `plots/ablation_tornado.png` | Factor impact tornado chart |
| `plots/scaling_configs.png` | Overhead vs max_steering_configs |
| `plots/scaling_batch.png` | Throughput vs batch_size |

**Total: ~5 min**

---

## 3-Machine Parallel Dispatch

For faster iteration, split across the 3 available RTX 3090 machines:

| Machine | Phases | Est. Time | Command |
|---------|--------|-----------|---------|
| A | 1 + 3 | 2.5 hr | `steering-bench run micro && steering-bench run ablation --model google/gemma-3-4b-it` |
| B | 2 | 2 hr | `steering-bench run vllm --model google/gemma-3-4b-it` |
| C | 4 | 1.5 hr | `steering-bench run external --model meta-llama/Llama-3.2-1B` |

Then collect results and run Phase 5 on any machine:

```bash
# Collect from all machines
rsync machineA:steering-bench/results/ results/machineA/
rsync machineB:steering-bench/results/ results/machineB/
rsync machineC:steering-bench/results/ results/machineC/

# Analyze
steering-bench analyze --results-dir results/ --output-dir results/reports/
```

**Single-machine canonical runs:** After development, rerun everything on one machine with pinned GPU clocks for the authoritative dataset. Tag with `--tag canonical`.

---

## Implementation Workstreams

For parallel agent development:

| Workstream | Files | Dependencies |
|-----------|-------|-------------|
| **WS1: Core** | `cli.py`, `config.py`, `vectors.py`, `timing.py`, `output.py`, `models.py`, `pyproject.toml`, `tests/` | None |
| **WS2: Micro** | `micro/steering_op.py`, `micro/steering_manager.py`, `micro/index_building.py` | WS1 (timing, output) |
| **WS3: vLLM** | `vllm_bench/latency.py`, `throughput.py`, `serving.py`, `memory.py`, `startup.py` | WS1 (all core) |
| **WS4: Ablation** | `ablation/cuda_graphs.py`, `compile_level.py`, `prefix_cache.py`, `config_scaling.py`, `hook_points.py`, `model_family.py` | WS1 + WS3 (reuses vllm_bench runners) |
| **WS5: External** | `external/base.py`, all 7 bench modules, `run_all.py` | WS1 (timing, output, vectors) |
| **WS6: Analysis** | `analysis/aggregate.py`, `plot_overhead.py`, `plot_comparison.py`, `plot_ablation.py`, `plot_scaling.py` | WS1 (output schema) |

**Dependency graph:**
```
WS1 (Core) ──┬── WS2 (Micro)
              ├── WS3 (vLLM) ── WS4 (Ablation)
              ├── WS5 (External)
              └── WS6 (Analysis)
```

WS2, WS3, WS5, WS6 can all start once WS1 is done. WS4 depends on WS3.

---

## Total Time Estimates

| Scenario | Wall-Clock |
|----------|-----------|
| Sequential on 1 machine | ~6.5 hours |
| Parallel on 3 machines | ~2.5 hours |
| Development iteration (single benchmark) | 10-30 min |
| Full canonical run (1 machine, all models) | ~8 hours |
