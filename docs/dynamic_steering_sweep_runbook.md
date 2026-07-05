# Dynamic-steering overhead sweep runbook (node0)

Three overhead sweeps for `google/gemma-3-4b-it` (34 decoder layers, hidden
2560) against the post-Wave-1 dynamic-steering vLLM branch, run on **node0**
(RTX 3090). The local box has **no GPU driver** — everything GPU runs on node0.

- Driver: `scripts/bench_dynamic_steering.py` (fresh subprocess per
  `(arm, batch_size)` cell; per-cell **arm-activation assertion** after warmup).
- Consumers: `src/steering_bench/capture_consumers/bench_consumers.py`.
- Presets: `scripts/run_dynamic_sweeps.sh {ladder|rowmon|sites|all}`.

## Arms

| arm | what it measures |
| --- | --- |
| `off` | baseline (no capture, no steering) |
| `capture_async` | async `on_capture` capture pipeline |
| `capture_sync` | per-step `on_step` capture pipeline |
| `steer_async` | global decode tier via the action queue |
| `steer_sync` | global decode tier via `on_step` |
| `steer_dynamic` | global in-graph monitor gating the tier (same-token) |
| `steer_override` | per-request dynamic-override pool (one row per request) |
| `steer_rowmon` | per-request override **+ per-row in-graph monitor** |

`(steer_rowmon − steer_override)` isolates the **per-row monitor** cost (never
previously benchmarked). `(steer_override − steer_sync)` isolates the
per-request override pool vs the single global tier.

## The three sweeps

```bash
# from the steering-bench repo root on node0, under the vllm-uv venv
bash scripts/run_dynamic_sweeps.sh ladder   # arms x bs {1,8,16,32}, post_block@17
bash scripts/run_dynamic_sweeps.sh rowmon   # off,override,rowmon x bs {1,8,16,32}
bash scripts/run_dynamic_sweeps.sh sites    # off,sync,override x bs 16 x site counts
bash scripts/run_dynamic_sweeps.sh all      # all three back to back
```

Results land under `results/dynamic_steering/{ladder,rowmon,sites}/`. Every
result records `vllm_commit`, `steer_layers`, `steer_hooks`, `num_sites`, the
monitor site, and the per-cell `arm_active` diagnostics.

The `sites` sweep runs these steered-site configs (monitor fixed at
`post_block@17`), overhead vs number of steered `(layer, hook)` sites:

| spec | hooks | sites |
| --- | --- | --- |
| `spread:1` | post_block | 1 |
| `spread:4` | post_block | 4 |
| `spread:8` | post_block | 8 |
| `spread:17` | post_block | 17 |
| `spread:34` | post_block | 34 |
| `spread:8` | pre_attn,post_attn,post_block | 24 |

`spread:N` = N evenly spaced layers across the 34-layer model (`spread:1` is the
middle layer, 17).

## Clocks (thermal transparency)

node0 = **10.1.1.69**, RTX 3090, clocks locked **1710 MHz**. Verify against the
per-cell recorded `gpu_clock_end_mhz` in each result. If not locked:

```bash
sudo nvidia-smi -lgc 1700     # lock before the sweep
# ... run sweeps ...
sudo nvidia-smi -rgc          # RESTORE clocks after
```

A clock drop across a sweep means throttling is confounding cross-cell deltas —
lock clocks or post-hoc rescale with the recorded clock.

## Deploy to node0

The local checkout (`/home/nymph/Code/vllm/dynamic-steering`) holds the built
`_C*.so`. Deploy the branch onto node0 by **rsync + editable-finder repoint**
(both are needed or submodules mix):

```bash
# 1. rsync the built vllm/ tree (contains _C*.so) to node0
rsync -a --delete /home/nymph/Code/vllm/dynamic-steering/vllm/ \
    node0:/home/nymph/Code/vllm/dynamic-steering/vllm/

# 2. repoint node0's editable finder mapping at the synced tree
#    (find the finder under the venv site-packages)
ssh node0 'f=$(ls /home/nymph/Code/vllm/*/.venv/lib/python3.12/site-packages/__editable___vllm_*_finder.py); \
  cp -n "$f" "$f.ORIG_BAK"; \
  sed -i "s#<old-vllm-path>#/home/nymph/Code/vllm/dynamic-steering#g" "$f"'
```

Keep the `.ORIG_BAK`. Editable finder repoint **and** the rsync are both
required — repointing without syncing runs stale code; syncing without
repointing loads the wrong submodules.

## Entry points must be visible

The steering-bench `vllm.capture_consumers` entry points must be resolvable in
the worker process. They must appear in **both**:

- the venv dist-info `entry_points.txt`, and
- `src/steering_bench.egg-info/entry_points.txt`.

After editing `pyproject.toml` entry points, refresh with
`uv pip install -e /home/nymph/Code/steering-bench` (or `pip install -e`) so the
new `bench_steer_override` / `bench_steer_rowmon` names register. Confirm:

```bash
python -c "from importlib.metadata import entry_points as ep; \
  print(sorted(e.name for e in ep(group='vllm.capture_consumers')))"
# expect: bench_capture_async bench_capture_sync bench_steer_async \
#         bench_steer_dynamic bench_steer_override bench_steer_rowmon bench_steer_sync
```

## Required environment

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process engine (fork+CUDA safe;
                                          # also lets the cell introspect the
                                          # worker for the activation assertion)
export VLLM_DISABLE_COMPILE_CACHE=1       # stale compile cache = wrong op arity
export VLLM_USE_FLASHINFER_SAMPLER=0      # node0 flashinfer sampler JIT fails
```

`run_dynamic_sweeps.sh` sets all three itself.

## Row-monitor engine flag (verified plumbing)

The `steer_rowmon` arm needs `enable_row_monitor` on the engine. The driver
passes `enable_row_monitor=True` as an `LLM(...)` kwarg; verified it reaches the
worker:

- `LLM(**kwargs)` forwards it into `EngineArgs.enable_row_monitor`
  (`vllm/engine/arg_utils.py:601`),
- which — when `enable_steering` is set — reaches
  `SteeringConfig(enable_row_monitor=...)` (`arg_utils.py:2219`),
- and is a `compute_hash` factor that resizes the per-row probe-table buffers
  (`vllm/config/steering.py:91`).

The per-cell activation assertion re-checks it actually landed via
`status["row_monitor"]["enabled"]` — so a silently dropped flag fails the cell
loudly (the offline `LLM(...)` override path has historically dropped such
bools).

## Caveats / re-run trigger

- vLLM PRs **#229 / #230 / #233** are still unmerged as of this writing.
  **#230 (warmup specialization fix)** is perf-relevant: **re-run `ladder`
  after #230 merges** and compare `vllm_commit` in the results. The recorded
  commit hash makes stale results obvious.
- Historical ladder shape to compare against: global-tier ~+1.5% <
  per-req ~+2.3% ≈ static ~+2.7% < async +5–6.7%, no regression past bs 16.
