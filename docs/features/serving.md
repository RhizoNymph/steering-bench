# serving (online/HTTP steering transport)

## Scope

The `ServingEngine` seam models the **online serving** axis: a long-lived
OpenAI-compatible API server driven by *streaming* completions whose value is the
per-token latency profile (TTFT / TPOT / ITL / E2EL) across steering modes. It is
a **separate ABC** from `SteeringEngine` because the transport (subprocess API
server + async streaming) and lifecycle (`start_server` / `stop_server`) cannot be
expressed by the synchronous `generate(list) -> list` contract.

**Non-scope:** offline batched generation (that is `SteeringEngine` + the sync
`Benchmark`), capture (Phase 4), and patch sweeps (Phase 6). The online *capture*
path (`bench_capture_serving.py`) is future work layered on this seam.

## Data / control flow

```
scripts/bench_serving.py  (thin driver)
  └─ VllmServingEngine(python_bin=…)              engine/engines/vllm_serving.py
  └─ ServingBenchmark(engine, BenchmarkConfig, **options).run()
       harness/benchmarks/serving.py
       │
       ├─ build prompts (synthetic | sharegpt) + typed SteeringSpec / NamedModuleRef per mode
       ├─ Phase 1 (if "disabled" in modes): start steering-OFF server, measure, stop
       └─ Phase 2 (steered modes): start ONE --enable-steering server (dev_mode if named_shared)
            for each mode:
              ├─ register_named_module(name, spec)        (named_shared only)
              ├─ warmup(warm_reqs) + drain barrier
              ├─ dump_and_reset_timings(mode, quiet=True) (discard warmup timings)
              ├─ run_workload(requests, concurrency)      → list[RequestResult]
              ├─ summarize_results(...)                   → TTFT/TPOT/ITL/E2EL block
              ├─ write_result(benchmark="vllm.serving", …, engine=identity())
              └─ dump_and_reset_timings(mode)             (print per-worker table)
```

Server start/stop are **synchronous** and run outside any event loop; each server
phase's measurement runs under a single `asyncio.run`. `start_server` health-polls
`GET /v1/models` until ready (or the process exits early).

## Payload encoding is owned by the adapter

Scripts never hand-encode wire bytes. `engine/serving.py` holds pure,
unit-testable functions:

- `pack_steering_vectors(spec)` — base64 little-endian float32 blob
  `{hook: {dtype, shape, layer_indices, data[, scales]}}` for the per-request
  `extra_body={"steering_vectors": …}` path. Rows are in ascending layer order;
  per-row `SteeringSpec.scales` ride along as a `scales` list.
- `steering_extra_body(steering)` — maps `None` / `SteeringSpec` / `NamedModuleRef`
  to the OpenAI `extra_body` dict (`steering_vectors` or `steering_name`).
- `named_register_payload(name, spec, prefill=, decode=)` — the
  `POST /v1/steering/modules/register` body `{name, vectors, prefill_vectors,
  decode_vectors}` (raw `{hook:{layer:[floats]}}` vectors — the register endpoint
  accepts raw, unlike the packed per-request field). Accepts a `SteeringSpec`
  (+ kwargs) or a `PhaseSteeringSpec` bundling all three.

## Metrics

`compute_request_metrics(t0, token_times, end_time)` builds a `RequestResult`
(`ttft_ms`, `itl_ms[]`, `e2el_ms`, `num_output_tokens`, `error`). `summarize_results`
aggregates the OK requests into the result block, with
`TPOT = (e2el - ttft) / max(1, num_output_tokens - 1)` per request and each family
reported via `timing.compute_stats` (minus raw samples). Formulas are identical to
the legacy `bench_serving.py` driver.

## SteeringSpec serving fields (the Phase-3 deferral)

- `SteeringSpec.scales: tuple[float,...] | None = None` — optional per-row
  multipliers; length must equal `num_vectors()` (canonical row order: hooks in
  insertion order, layers ascending). `scales_for(hook)` returns the per-hook
  slice; `with_scales(...)` returns a validated copy. Defaults `None` → behaves
  exactly as pre-serving.
- `PhaseSteeringSpec(base, prefill=None, decode=None)` — companion type bundling
  phase-split variants for `named_register_payload`.

## Related files

| File | Role |
|------|------|
| `src/steering_bench/engine/serving.py` | `ServingEngine` ABC, `ServingConfig`, `RequestResult`, metrics, adapter-owned encoders, serving-engine registry. |
| `src/steering_bench/engine/engines/vllm_serving.py` | `VllmServingEngine` — subprocess launch (`build_server_command`/`build_server_env`), `AsyncOpenAI` streaming driver, register/timing HTTP calls. Lazy `openai`/`httpx`/`vllm`. |
| `src/steering_bench/harness/benchmarks/serving.py` | `ServingBenchmark` orchestrator + serving mode catalog (`build_serving_requests`, `shared_spec_for`, `diverse_specs_for`, prompt loaders). |
| `src/steering_bench/engine/spec.py` | `SteeringSpec.scales`, `PhaseSteeringSpec`. |
| `src/steering_bench/engine/base.py` | `Capabilities.serving` flag. |
| `scripts/bench_serving.py` | Thin driver → `VllmServingEngine` + `ServingBenchmark`. |

## Invariants / constraints

- `ServingEngine` is separate from `SteeringEngine`; a serving adapter advertises
  `Capabilities(serving=True)` and lives in `SERVING_ENGINE_REGISTRY`, not the
  `SteeringEngine` registry.
- `VllmServingEngine` runs at most one server at a time (`start_server` raises if
  one is live); `stop_server` graceful-then-hard-kills the process group.
- The register + timing-dump endpoints require the server started with
  `dev_mode` (`VLLM_SERVER_DEV_MODE=1`) / `timing` (`VLLM_STEERING_TIMING=1`);
  `dump_and_reset_timings` no-ops (404 / connect error) otherwise.
- `disabled` mode requires its own steering-off server; steered modes share one.
- Encoders (`pack_steering_vectors`, `steering_extra_body`, `named_register_payload`,
  `compute_request_metrics`, `summarize_results`) are pure — no vllm/openai/httpx.

## Verification (CPU / no live server)

- `tests/test_spec_serving_fields.py`, `test_pack_steering_vectors.py`,
  `test_serving_metrics.py`, `test_register_payload.py`, `test_harness_serving.py`.
- `python -m steering_bench list` shows `serving`; `run serving --help`;
  `python scripts/bench_serving.py --help`.
- GPU + a running fork server (`--enable-steering`, `VLLM_SERVER_DEV_MODE=1`) is
  the user's acceptance step.
