"""Engine-agnostic capture-overhead benchmark (Tier 1b).

Measures the wall-clock ``generate()`` overhead of enabling capture consumers,
expressed over the CaptureEngine seam so it selects only capture-capable engines
(``required_capabilities = Capabilities(capture=True)``). Replaces the inline
``LLM(capture_consumers=[...])`` + hand-rolled config dicts in
``scripts/bench_capture_e2e.py`` with typed :class:`CaptureConsumerSpec` /
:class:`RequestCapture`.

Config catalog (the seam-expressible arms of the e2e script):

- ``baseline``           -- no consumers (reference).
- ``logging_minimal``    -- 1 logging consumer, ``last_prompt``, one layer.
- ``logging_max``        -- 1 logging consumer, ``all`` positions, all layers.
- ``logging_3x``         -- 3 logging consumers on one hook/layer (union-gather).
- ``filesystem_minimal`` -- 1 filesystem consumer + a per-request capture opt-in
  (the filesystem consumer has ``reads_client_spec=True``).
- ``driver_minimal``     -- 1 driver ``RecordingDriverConsumer`` instance.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, ClassVar

from steering_bench.engine.base import Capabilities
from steering_bench.engine.capture import CaptureConsumerSpec
from steering_bench.engine.spec import GenerationRequest, RequestCapture
from steering_bench.harness.benchmark import Benchmark
from steering_bench.harness.benchmarks.workload import make_prompt
from steering_bench.harness.models import get_model_config

CAPTURE_CONFIGS: tuple[str, ...] = (
    "baseline",
    "logging_minimal",
    "logging_max",
    "logging_3x",
    "filesystem_minimal",
    "driver_minimal",
)
DEFAULT_CONFIG = "logging_minimal"
DEFAULT_PROMPT_LEN = 64


def _mid_layer(model: str, layer: int) -> int:
    """Clamp the requested layer to the model's layer count."""
    cfg = get_model_config(model)
    return min(layer, cfg.num_layers - 1)


def build_consumer_specs(
    config: str,
    *,
    model: str,
    layer: int,
    hook: str,
    fs_root: str | None = None,
    driver_instance: Any | None = None,
) -> list[CaptureConsumerSpec]:
    """Build the capture-consumer spec list for ``config``.

    Pure for the logging / filesystem configs. ``driver_minimal`` needs a
    pre-built ``driver_instance`` (a fork-backed ``RecordingDriverConsumer``);
    ``filesystem_minimal`` needs a writable ``fs_root``.
    """
    cfg = get_model_config(model)
    mid = min(layer, cfg.num_layers - 1)
    all_layers = list(range(cfg.num_layers))
    match config:
        case "baseline":
            return []
        case "logging_minimal":
            return [
                CaptureConsumerSpec(
                    name="logging",
                    params={"hooks": {hook: [mid]}, "positions": "last_prompt", "level": "WARNING"},
                )
            ]
        case "logging_max":
            return [
                CaptureConsumerSpec(
                    name="logging",
                    params={"hooks": {hook: all_layers}, "positions": "all", "level": "WARNING"},
                )
            ]
        case "logging_3x":
            return [
                CaptureConsumerSpec(
                    name="logging",
                    instance_name=f"log_{c}",
                    params={"hooks": {hook: [mid]}, "positions": "last_prompt", "level": "WARNING"},
                )
                for c in ("a", "b", "c")
            ]
        case "filesystem_minimal":
            if fs_root is None:
                raise ValueError("filesystem_minimal requires an fs_root")
            return [
                CaptureConsumerSpec(
                    name="filesystem",
                    params={"root": fs_root, "writer_threads": 4},
                )
            ]
        case "driver_minimal":
            if driver_instance is None:
                raise ValueError("driver_minimal requires a driver_instance")
            return [
                CaptureConsumerSpec(
                    name="driver_recording",
                    location="driver",
                    instance=driver_instance,
                )
            ]
    raise ValueError(f"unknown capture config {config!r}; known: {CAPTURE_CONFIGS}")


def build_request_capture(
    config: str, *, model: str, layer: int, hook: str
) -> RequestCapture | None:
    """The per-request capture opt-in for ``config`` (only ``filesystem_minimal``)."""
    if config != "filesystem_minimal":
        return None
    return RequestCapture(
        consumer="filesystem",
        hooks={hook: [_mid_layer(model, layer)]},
        positions="last_prompt",
        tag="benchmark",
    )


class CaptureBenchmark(Benchmark):
    """Capture-overhead benchmark: generate() throughput with consumers enabled."""

    benchmark_name: ClassVar[str] = "capture.e2e"
    required_capabilities: ClassVar[Capabilities] = Capabilities(capture=True)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--capture-config",
            default=DEFAULT_CONFIG,
            choices=CAPTURE_CONFIGS,
            help="Which capture-consumer configuration to measure.",
        )
        parser.add_argument(
            "--batch-size", type=int, default=1,
            help="Requests generated per measured iteration.",
        )
        parser.add_argument(
            "--prompt-len", type=int, default=DEFAULT_PROMPT_LEN,
            help="Approximate prompt length in tokens.",
        )

    @staticmethod
    def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "capture_config": args.capture_config,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
        }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._fs_root: str | None = None

    # -- convenience accessors ----------------------------------------------

    @property
    def capture_config(self) -> str:
        return str(self.options.get("capture_config", DEFAULT_CONFIG))

    @property
    def batch_size(self) -> int:
        return int(self.options.get("batch_size", 1))

    @property
    def prompt_len(self) -> int:
        return int(self.options.get("prompt_len", DEFAULT_PROMPT_LEN))

    # -- Benchmark hooks -----------------------------------------------------

    def _driver_instance(self) -> Any:
        from steering_bench.capture_consumers.consumers import RecordingDriverConsumer

        return RecordingDriverConsumer(
            hooks={self.config.hook: [_mid_layer(self.config.model, self.config.layer)]},
            positions="last_prompt",
        )

    def before_load(self) -> None:
        if self.capture_config == "filesystem_minimal":
            self._tmp = tempfile.TemporaryDirectory(prefix="bench-capture-")
            self._fs_root = self._tmp.name
        driver_instance = (
            self._driver_instance() if self.capture_config == "driver_minimal" else None
        )
        specs = build_consumer_specs(
            self.capture_config,
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
            fs_root=self._fs_root,
            driver_instance=driver_instance,
        )
        self.engine.configure_capture(specs)

    def after_teardown(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
            self._fs_root = None

    def build_requests(self) -> list[GenerationRequest]:
        prompt = make_prompt(self.prompt_len)
        capture = build_request_capture(
            self.capture_config,
            model=self.config.model,
            layer=self.config.layer,
            hook=self.config.hook,
        )
        return [
            GenerationRequest(
                prompt=prompt, max_tokens=self.config.max_tokens, capture=capture
            )
            for _ in range(self.batch_size)
        ]

    def parameters(self) -> dict[str, Any]:
        params = super().parameters()
        params["capture_config"] = self.capture_config
        params["batch_size"] = self.batch_size
        params["prompt_len"] = self.prompt_len
        return params
