# vLLM-internal microbenchmarks

These benchmarks exercise **vLLM-fork internals directly** — the raw steering
primitives (op kernel, `SteeringManager`, index/hash construction) and the
capture-manager hot path (`vllm.v1.capture.*`) — with **no model load** and **no
cross-engine analog**.

They are intentionally **outside** the cross-engine framework
(`steering_bench.engine` / `steering_bench.harness`): there is no
`SteeringEngine` / `Benchmark` seam they could run through, and they are **not
expected to run on any non-vLLM engine**. Later migration phases and the Tier-3
sweep should leave this directory alone — do **not** re-plumb these onto the
engine seam.

Each script still writes results via `steering_bench.output.write_result` and
resolves the package by inserting `../../src` on `sys.path`. They require the
vLLM fork (and, for most, CUDA) to actually run; on a plain box they either fall
back to a reference impl or fail at the `import vllm` / CUDA check.

## Scripts

| Script | Fork internal it measures |
|--------|---------------------------|
| `bench_steering_op.py` | Latency of the steering op kernel (`torch.ops.vllm.apply_steering`, or a reference impl when the custom op is unregistered) across hidden size × tokens × table rows × dtype. |
| `bench_steering_manager.py` | Python-side overhead of `vllm.v1.worker.steering_manager.SteeringManager` (`register_config`/`release_config`, `populate_steering_tables`, `get_row_for_config`) across layer and config counts. |
| `bench_index_building.py` | The CPU loop that fills the `steering_index` tensor (token position → steering table row), uniform and mixed prefill/decode phases. |
| `bench_hash.py` | Cost of `vllm.config.steering_types.hash_steering_config` on a representative steering-vector dict. |
| `bench_capture_manager.py` | The three phases of the `vllm.v1.capture.manager.CaptureManager` hot path — `build_step_plan` (CPU), `on_hook` (GPU gather), `dispatch_step_captures` (fan-out + GPU→CPU copy) — with `NullCaptureSink` (no I/O). |
| `bench_capture_latency.py` | Capture delivery latency: `--mode microbench` drives `CaptureManager` with a timestamping sink (dispatch-added delay only); `--mode e2e` runs `LLM.generate()` with a driver-side consumer. |
| `bench_capture_plugin_work.py` | How much CPU a capture plugin can burn per chunk before dispatch/throughput degrade — sweeps per-chunk work cost (busy/sleep/queue) against `CaptureManager` (microbench) or `LLM.generate()` (e2e). |
| `bench_capture_packed.py` | `per_file` vs `packed` on-disk layout throughput driven through the full `vllm.v1.capture.consumers.filesystem.FilesystemConsumer` (metadata-RPC savings from packing per-request). |

Note: `bench_hash.py` is a bare timing script (no argparse) and imports the fork
at module import — it has no `--help` and runs its benchmark on execution.
