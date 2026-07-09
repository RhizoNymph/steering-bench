"""vLLM online-serving adapter for the :class:`ServingEngine` seam.

Wraps the fork's OpenAI-compatible API server (``vllm.entrypoints.openai.api_server``)
launched as a subprocess, drives it through ``AsyncOpenAI`` streaming completions,
and owns the steering wire encoding (inline packed vectors + named-module
register payload + timing dump) so scripts never hand-encode.

The HTTP stack (``openai`` / ``httpx``) is imported LAZILY inside method bodies,
mirroring the other adapters, so this module imports without vllm/openai/httpx.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from steering_bench.engine.base import Capabilities
from steering_bench.engine.serving import (
    RequestResult,
    ServingConfig,
    ServingEngine,
    ServingError,
    compute_request_metrics,
    named_register_payload,
    steering_extra_body,
)
from steering_bench.engine.spec import (
    GenerationRequest,
    PhaseSteeringSpec,
    SteeringSpec,
)


def build_server_command(
    python_bin: str, model_id: str, config: ServingConfig
) -> list[str]:
    """Assemble the ``python -m vllm.entrypoints.openai.api_server`` argv. Pure."""
    cmd = [
        python_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    if config.enable_steering:
        cmd += ["--enable-steering", "--max-steering-configs", str(config.max_steering_configs)]
    if config.enforce_eager:
        cmd.append("--enforce-eager")
    cmd += list(config.extra_flags)
    return cmd


def build_server_env(config: ServingConfig) -> dict[str, str]:
    """The subprocess environment for the server (dev-mode + timing toggles). Pure."""
    env = os.environ.copy()
    if config.dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if config.timing:
        env["VLLM_STEERING_TIMING"] = "1"
    return env


class VllmServingEngine(ServingEngine):
    """Serving adapter over the vLLM fork's OpenAI API server."""

    name = "vllm"
    capabilities = Capabilities(serving=True, named_modules=True, config_capacity=True)

    def __init__(
        self,
        *,
        python_bin: str | None = None,
        log_dir: str | Path = "/tmp",
    ) -> None:
        self._python_bin = python_bin or sys.executable
        self._log_dir = Path(log_dir)
        self._proc: subprocess.Popen | None = None
        self._config: ServingConfig | None = None
        self._model_id: str | None = None

    # -- lifecycle -----------------------------------------------------------

    def start_server(self, model_id: str, *, config: ServingConfig) -> None:
        if self._proc is not None:
            raise ServingError("start_server called while a server is already running")
        self._config = config
        self._model_id = model_id
        self._log_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_server_command(self._python_bin, model_id, config)
        env = build_server_env(config)
        log_path = self._log_dir / f"vllm_serving_{config.port}.log"
        print(f"[server] launch: {' '.join(cmd)}")
        print(f"[server] log:    {log_path}")
        log_f = open(log_path, "wb")
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )
        asyncio.run(self._wait_for_server(config.startup_timeout))

    async def _wait_for_server(self, timeout: float) -> None:
        import httpx

        base = self.base_url
        deadline = time.perf_counter() + timeout
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.perf_counter() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise ServingError(
                        f"server process exited early (code {self._proc.returncode})"
                    )
                try:
                    r = await client.get(f"{base}/models")
                    if r.status_code == 200:
                        print(f"[server] ready at {base}")
                        return
                except Exception as e:  # noqa: BLE001 - poll until healthy
                    last_err = e
                await asyncio.sleep(2.0)
        raise ServingError(
            f"server {base} not ready within {timeout}s (last error: {last_err})"
        )

    def stop_server(self, grace: float = 15.0) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        print(f"[server] terminate pid={proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        print(f"[server] kill -9 pid={proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            print(f"[server] WARNING: pid={proc.pid} did not die")

    @property
    def base_url(self) -> str:
        if self._config is None:
            raise ServingError("base_url read before start_server")
        return f"http://{self._config.host}:{self._config.port}/v1"

    # -- async streaming driver ----------------------------------------------

    async def run_request(self, request: GenerationRequest) -> RequestResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key="unused")
        extra_body = steering_extra_body(request.steering)
        token_times: list[float] = []
        try:
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": 0.0,
                "stream": True,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            t0 = time.perf_counter()
            stream = await client.completions.create(**kwargs)
            async for chunk in stream:
                now = time.perf_counter()
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].text
                if not delta:
                    continue
                token_times.append(now)
            end = time.perf_counter()
            return compute_request_metrics(t0, token_times, end_time=end)
        except Exception as e:  # noqa: BLE001 - per-request isolation
            return RequestResult(error=f"{type(e).__name__}: {e}")

    # -- admin endpoints (dev-mode) ------------------------------------------

    async def register_named_module(
        self,
        name: str,
        spec: SteeringSpec | PhaseSteeringSpec,
        *,
        prefill: SteeringSpec | None = None,
        decode: SteeringSpec | None = None,
        timeout: float = 60.0,
    ) -> None:
        import httpx

        payload = named_register_payload(name, spec, prefill=prefill, decode=decode)
        endpoint = f"{self._admin_root()}/v1/steering/modules/register"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(endpoint, json=payload)
            if r.status_code != 200:
                raise ServingError(
                    f"register_named_module(name={name}) failed: {r.status_code} {r.text}"
                )
        print(f"[server] registered named module: {name}")

    async def dump_and_reset_timings(
        self, mode: str, *, quiet: bool = False, timeout: float = 30.0
    ) -> None:
        import httpx

        endpoint = f"{self._admin_root()}/v1/steering/_timings/dump_and_reset"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(endpoint)
        except Exception as e:  # noqa: BLE001 - endpoint may be absent
            if not quiet:
                print(f"[timing] dump endpoint unavailable ({e}) - skipping")
            return
        if r.status_code == 404:
            return
        if r.status_code != 200:
            if not quiet:
                print(f"[timing] dump endpoint returned {r.status_code}: {r.text[:200]}")
            return
        if quiet:
            return
        self._print_timings(mode, r.json().get("workers", []))

    def _admin_root(self) -> str:
        return self.base_url.rstrip("/").removesuffix("/v1")

    @staticmethod
    def _print_timings(mode: str, workers: list) -> None:
        if not workers:
            return
        print(f"\n[timing mode={mode}] per-worker steering breakdown")
        for i, worker in enumerate(workers):
            if not worker:
                continue
            name_w = max(len(row[0]) for row in worker)
            print(
                f"  worker[{i}]  {'name':<{name_w}}  {'n':>8}  {'total_ms':>12}  "
                f"{'mean_us':>10}  {'max_ms':>10}"
            )
            for name, count, total_ns, max_ns in worker:
                total_ms = total_ns / 1e6
                mean_us = total_ns / count / 1e3 if count else 0.0
                print(
                    f"  worker[{i}]  {name:<{name_w}}  {count:>8d}  {total_ms:>12.3f}  "
                    f"{mean_us:>10.2f}  {max_ns / 1e6:>10.3f}"
                )

    # -- identity ------------------------------------------------------------

    def version(self) -> str:
        import vllm

        return vllm.__version__

    def commit(self) -> str | None:
        import vllm

        return getattr(vllm, "__commit__", None)
