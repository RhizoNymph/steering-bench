# patch_sweep

Activation-patching / causal-tracing brought into the framework as a **third,
distinct axis** — not steering-generate, not capture, not online serving.

## Scope

- A `PatchSweepEngine` ABC modelling one denoising sweep: patch the clean run's
  residual into the corrupt run at every `(layer, position)` cell and grade the
  answer token's recovered logprob.
- A typed `PatchSweepResult` (the UNION of both backends' reported fields), a
  `PatchSweepRequest` input, a `PatchSweepArgmax` reduction, and pure
  dict→result mappers.
- A `discover_patch_sweep` registry with two adapters (TransformerLens in-process
  + vLLM over HTTP) wrapping the already-verified `external/` modules.
- A `PatchSweepBenchmark` comparison orchestrator + `run patch-sweep` CLI, and a
  thin `scripts/bench_patching_external.py` driver over it.

## Non-scope

- The patching mechanics themselves. The verified logic stays in
  `external/tl_patching.py` (`load_model` / `run_patch_sweep`) and
  `external/vllm_patch_sweep.py` (`server_healthy` / `run_patch_sweep`); the
  adapters import and map, they do not reimplement.
- Generation latency / streaming metrics (those are the `Benchmark` /
  `ServingEngine` axes).

## Running the vLLM sweep (server requirements)

Unlike the serving axis, the vLLM patch-sweep adapter does **not** launch its own
server — it drives an existing one via `--base-url` (bare host or `/v1`-suffixed;
both normalized). That server must be launched with **all three** flags:

```
--enable-patching                 # mounts the /v1/patch_sweep route
--capture-consumers patch_source  # the consumer that stores clean-run activations
--patch-source-cache-bytes <N>    # allocates the source store, e.g. 2000000000
```

The client uses the server's **one-call auto-capture** path (it sends
`clean_prompt` + a fresh `source_run`), so no separate capture step is needed —
but the auto-captured rows are written through the `patch_source` consumer into
the source store. **If `--patch-source-cache-bytes` is omitted the store does not
exist, captured rows are silently dropped, and the sweep 400s with `patch source
not found`.** `run_patch_sweep` surfaces the server's error body plus this
flag hint (via `PatchSweepServerError`, rewrapped as `PatchSweepError`), so a
misconfigured server is diagnosable instead of an opaque HTTP 400.

Validated end-to-end on an RTX 3090 against the fork's `feat/integration` build
(`gemma-3-4b-it`): `run patch-sweep --variants vllm_sweep --base-url
http://host:8765/v1 --num-layers 12` → 72 cells at ~59 cells/s, argmax L4@4.

## Result shape (`PatchSweepResult`)

Common fields (every backend): `variant`, `cells`, `n_layers`, `n_positions`,
`wall_s`, `cells_per_s`, `clean_logprob`, `corrupt_logprob`, `argmax`
(`PatchSweepArgmax{layer, position, recovered}`).

vLLM-only fields (default `None` for TransformerLens): `noise_floor`,
`auto_captured`, `skipped`.

`to_dict()` emits the common fields always and the vLLM-only fields only when
non-`None`, so a TransformerLens result serializes to exactly the dict
`tl_patching.run_patch_sweep` produced and a vLLM result to exactly the dict
`vllm_patch_sweep.run_patch_sweep` produced (round-trip verified in tests).

## Control / data flow

```
run patch-sweep  ─▶  PatchSweepBenchmark.run()
   │  parse --variants → (tl_variants, want_vllm)
   │  discover_patch_sweep(filter_names=[...])       # prints availability/skips
   ├─ tl:   TLPatchSweepEngine.setup(model, dtype)
   │        for prompt × variant: run_sweep(PatchSweepRequest)
   │            → external.tl_patching.run_patch_sweep(...) → from_tl_dict
   │        teardown(); write_result(benchmark="tl.patch_sweep", engine=identity)
   ├─ vllm: VllmPatchSweepEngine.setup(base_url)      # server_healthy check
   │        for prompt × rep: run_sweep(PatchSweepRequest)
   │            → external.vllm_patch_sweep.run_patch_sweep(...) → from_vllm_dict
   │        teardown(); write_result(benchmark="vllm.patch_sweep", engine=identity)
   └─ cross-tool argmax-agreement check + cells/s summary (printed)
```

Each engine's runs are written to their own result file stamped with that
engine's `identity()` block (mirroring how `ServingBenchmark` writes per-mode).
The comparison output the legacy script produced — the argmax-position
disagreement warnings and the per-prompt cells/s summary — is preserved.

## Availability / discovery

- **tl** (`required_package="transformer_lens"`): skipped with a printed reason
  when the package is absent.
- **vllm** (`required_package=None`, `needs_base_url=True`): always importable
  (httpx is imported lazily inside the wrapped module), so it is always listed by
  discovery — annotated "requires a reachable --base-url server". Real
  availability is a healthy server, checked in `setup` via `server_healthy`;
  failure raises `PatchSweepError`, which the benchmark catches and reports as a
  SKIP.

## Files

- `src/steering_bench/engine/patch_sweep.py` — the ABC, `PatchSweepResult` /
  `PatchSweepArgmax` / `PatchSweepRequest`, `PatchSweepError`, the registry, and
  `discover_patch_sweep` / `get_patch_sweep_engine`.
- `src/steering_bench/engine/engines/patch_sweep_tl.py` — `TLPatchSweepEngine`
  (naive + batched; wraps `external/tl_patching.py`).
- `src/steering_bench/engine/engines/patch_sweep_vllm.py` — `VllmPatchSweepEngine`
  (wraps `external/vllm_patch_sweep.py`).
- `src/steering_bench/harness/benchmarks/patch_sweep.py` — `PatchSweepBenchmark`,
  prompt pairs, variant/prompt parsing, per-engine writing + summary.
- `src/steering_bench/harness/benchmarks/registry.py` — `PATCH_SWEEP_REGISTRY` +
  `get_patch_sweep_benchmark`.
- `src/steering_bench/__main__.py` — `_run_patch_sweep` dispatch + `list` entry.
- `scripts/bench_patching_external.py` — thin driver over the benchmark.
- `src/steering_bench/external/{tl_patching,vllm_patch_sweep}.py` — the verified
  mechanisms (unchanged, imported by the adapters).

## Invariants / constraints

- The seam modules import no heavy backend at import scope; `transformer_lens` /
  `httpx` / `vllm` are imported lazily inside method bodies.
- `PatchSweepResult` / `PatchSweepArgmax` / `PatchSweepRequest` are frozen.
- The dict→result mappers are pure and total over the documented backend dict
  shapes (including a null argmax), testable without any backend installed.
- `Capabilities.patch_sweep` is additive; both adapters advertise it.
- The patch-sweep axis is independent: it does not touch the sync `Benchmark`
  measure-loop or the `ServingEngine` path.
