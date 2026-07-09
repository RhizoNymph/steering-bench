"""PatchSweepEngine seam: activation-patching / causal-tracing as a third axis.

The synchronous ``SteeringEngine.generate`` ABC and the streaming
``ServingEngine`` both measure token *generation*. Causal tracing is a different
study entirely: patch the clean run's residual into the corrupt run at every
``(layer, position)`` cell and grade the answer token's recovered logprob. Its
result is a whole grid reduced to ``cells / wall_s / cells_per_s`` and an argmax
cell -- not a latency profile -- so it gets its own ABC, its own typed result,
and its own small registry, mirroring the SEPARATE-ABC pattern used for serving.

Two impls already exist under :mod:`steering_bench.external`:

* :mod:`steering_bench.external.tl_patching` -- TransformerLens in-process,
  ``naive`` (one forward per cell) and ``batched`` (one forward per layer)
  variants.
* :mod:`steering_bench.external.vllm_patch_sweep` -- the fork's one-call
  ``POST /v1/patch_sweep`` over HTTP (adds server-side auto-capture, a batch
  noise floor, and source cleanup).

Both return a plain ``dict``; :class:`PatchSweepResult` is the typed UNION of
their fields (vLLM-only fields default ``None`` for the TransformerLens impl).
Nothing in this module imports ``transformer_lens`` / ``httpx`` / ``vllm`` at
import scope: the ABC and the dict->result mappers are pure. Concrete adapters
(:mod:`steering_bench.engine.engines.patch_sweep_tl` /
:mod:`steering_bench.engine.engines.patch_sweep_vllm`) import their backend
lazily inside method bodies.
"""

from __future__ import annotations

import abc
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

from steering_bench.engine.base import Capabilities


class PatchSweepError(RuntimeError):
    """Raised when a patch-sweep engine is misused or its backend is unavailable."""


# -- typed result ------------------------------------------------------------


@dataclass(frozen=True)
class PatchSweepArgmax:
    """The recovered-metric argmax cell of a sweep.

    Fields are ``None`` when a backend reports no argmax (e.g. an empty vLLM
    grid); otherwise ``layer`` / ``position`` locate the causal site and
    ``recovered`` is its recovered-metric value.
    """

    layer: int | None
    position: int | None
    recovered: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "position": self.position,
            "recovered": self.recovered,
        }


@dataclass(frozen=True)
class PatchSweepResult:
    """One completed patch sweep, the UNION of both backends' reported fields.

    The common fields (``variant`` .. ``argmax``) are produced by every backend.
    ``noise_floor`` / ``auto_captured`` / ``skipped`` are vLLM-only and default
    to ``None`` for the TransformerLens variants, so a single type describes both
    and :meth:`to_dict` reproduces each backend's original dict shape (the
    vLLM-only keys are emitted only when present).
    """

    variant: str
    cells: int
    n_layers: int
    n_positions: int
    wall_s: float
    cells_per_s: float
    clean_logprob: float | None
    corrupt_logprob: float | None
    argmax: PatchSweepArgmax
    #: vLLM-only: batch-noise-floor rerun logprob.
    noise_floor: float | None = None
    #: vLLM-only: whether the server auto-captured the clean run.
    auto_captured: bool | None = None
    #: vLLM-only: count of skipped cells (server-side).
    skipped: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the result-writing dict, reproducing the source shape.

        The common fields always appear; the three vLLM-only fields appear only
        when non-``None`` so a TransformerLens result serializes to exactly the
        dict :mod:`steering_bench.external.tl_patching` produced.
        """
        d: dict[str, Any] = {
            "variant": self.variant,
            "cells": self.cells,
            "n_layers": self.n_layers,
            "n_positions": self.n_positions,
            "wall_s": self.wall_s,
            "cells_per_s": self.cells_per_s,
            "clean_logprob": self.clean_logprob,
            "corrupt_logprob": self.corrupt_logprob,
            "argmax": self.argmax.to_dict(),
        }
        if self.noise_floor is not None:
            d["noise_floor"] = self.noise_floor
        if self.auto_captured is not None:
            d["auto_captured"] = self.auto_captured
        if self.skipped is not None:
            d["skipped"] = self.skipped
        return d

    # -- dict -> result mappers (pure; no heavy imports) ---------------------

    @classmethod
    def from_tl_dict(cls, d: dict[str, Any]) -> PatchSweepResult:
        """Map a :func:`tl_patching.run_patch_sweep` dict to a typed result.

        The TransformerLens dict has no vLLM-only fields; they default to
        ``None``. The ``argmax`` sub-dict is reduced to a
        :class:`PatchSweepArgmax`.
        """
        a = d["argmax"]
        return cls(
            variant=d["variant"],
            cells=d["cells"],
            n_layers=d["n_layers"],
            n_positions=d["n_positions"],
            wall_s=d["wall_s"],
            cells_per_s=d["cells_per_s"],
            clean_logprob=d.get("clean_logprob"),
            corrupt_logprob=d.get("corrupt_logprob"),
            argmax=PatchSweepArgmax(
                layer=a.get("layer"),
                position=a.get("position"),
                recovered=a.get("recovered"),
            ),
        )

    @classmethod
    def from_vllm_dict(cls, d: dict[str, Any]) -> PatchSweepResult:
        """Map a :func:`vllm_patch_sweep.run_patch_sweep` dict to a typed result.

        Carries the three vLLM-only fields (``noise_floor`` / ``auto_captured`` /
        ``skipped``) through in addition to the common fields.
        """
        a = d["argmax"]
        return cls(
            variant=d["variant"],
            cells=d["cells"],
            n_layers=d["n_layers"],
            n_positions=d["n_positions"],
            wall_s=d["wall_s"],
            cells_per_s=d["cells_per_s"],
            clean_logprob=d.get("clean_logprob"),
            corrupt_logprob=d.get("corrupt_logprob"),
            argmax=PatchSweepArgmax(
                layer=a.get("layer"),
                position=a.get("position"),
                recovered=a.get("recovered"),
            ),
            noise_floor=d.get("noise_floor"),
            auto_captured=d.get("auto_captured"),
            skipped=d.get("skipped"),
        )


# -- request -----------------------------------------------------------------


@dataclass(frozen=True)
class PatchSweepRequest:
    """Inputs for one patch sweep: the clean/corrupt prompt pair + answer token.

    ``variant`` selects the TransformerLens mechanism (``naive`` | ``batched``);
    the vLLM adapter ignores it (the server picks its own path). ``n_layers`` is
    the layer count the vLLM adapter sweeps (``layers = {start:0, stop:n_layers}``);
    the TransformerLens adapter reads the count off the loaded model and ignores
    it. ``logits_chunk_budget`` bounds the TransformerLens ``batched`` logits
    materialization.
    """

    clean: str
    corrupt: str
    answer: str
    variant: str = "batched"
    n_layers: int = 28
    logits_chunk_budget: int = 4096

    def __post_init__(self) -> None:
        for name in ("clean", "corrupt", "answer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PatchSweepError(
                    f"PatchSweepRequest.{name} must be a non-empty str, got {value!r}"
                )
        if self.n_layers < 1:
            raise PatchSweepError(
                f"PatchSweepRequest.n_layers must be >= 1, got {self.n_layers}"
            )
        if self.logits_chunk_budget < 1:
            raise PatchSweepError(
                "PatchSweepRequest.logits_chunk_budget must be >= 1, got "
                f"{self.logits_chunk_budget}"
            )


# -- the PatchSweepEngine ABC ------------------------------------------------


class PatchSweepEngine(abc.ABC):
    """Abstract base for an activation-patching / causal-tracing adapter.

    Distinct from :class:`~steering_bench.engine.base.SteeringEngine` and
    :class:`~steering_bench.engine.serving.ServingEngine`: the measured quantity
    is a ``(layers x positions)`` denoising grid reduced to throughput + an
    argmax cell, not generation latency. Adapters set ``name`` / ``capabilities``
    (with ``Capabilities.patch_sweep = True``) and implement the tiny lifecycle:
    ``setup`` -> ``run_sweep`` -> ``teardown``. Heavy backends
    (``transformer_lens`` / ``httpx``) are imported lazily inside method bodies.
    """

    name: str = "unknown"
    capabilities: Capabilities = Capabilities(patch_sweep=True)

    @abc.abstractmethod
    def setup(self, model_id: str, **opts: Any) -> None:
        """Prepare the backend for sweeps.

        The TransformerLens adapter loads ``model_id`` in-process; the vLLM
        adapter records the ``base_url`` (from ``opts``) and health-checks the
        already-running server (``model_id`` is only metadata there).
        """

    @abc.abstractmethod
    def run_sweep(self, request: PatchSweepRequest) -> PatchSweepResult:
        """Run one full sweep for ``request`` and return the typed result."""

    def teardown(self) -> None:
        """Release backend resources. Default: no-op."""

    # -- identity (overridden by adapters) -----------------------------------

    def version(self) -> str:
        return "unknown"

    def commit(self) -> str | None:
        return None

    def identity(self) -> dict[str, str | None]:
        return {"name": self.name, "version": self.version(), "commit": self.commit()}


# -- patch-sweep registry ----------------------------------------------------


@dataclass(frozen=True)
class PatchSweepEngineEntry:
    """A registered patch-sweep adapter, described without importing it.

    ``required_package`` gates discovery on import availability (the
    TransformerLens adapter needs ``transformer_lens``). ``needs_base_url`` flags
    an adapter whose real availability is a *reachable server* rather than a
    Python package: it is always importable (``required_package=None``) but
    ``setup`` fails without a healthy ``--base-url`` server.
    """

    name: str
    required_package: str | None
    module_path: str
    class_name: str
    needs_base_url: bool = False


PATCH_SWEEP_REGISTRY: list[PatchSweepEngineEntry] = [
    PatchSweepEngineEntry(
        name="tl",
        required_package="transformer_lens",
        module_path="steering_bench.engine.engines.patch_sweep_tl",
        class_name="TLPatchSweepEngine",
    ),
    PatchSweepEngineEntry(
        name="vllm",
        required_package=None,
        module_path="steering_bench.engine.engines.patch_sweep_vllm",
        class_name="VllmPatchSweepEngine",
        needs_base_url=True,
    ),
]


def is_package_available(package: str | None) -> bool:
    """Whether ``package`` is importable. ``None`` means no package dependency."""
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None


def discover_patch_sweep(
    filter_names: list[str] | None = None,
) -> list[type[PatchSweepEngine]]:
    """Return patch-sweep adapter classes usable in this environment.

    An adapter is skipped (with a printed reason) when it is filtered out, when
    its required package is missing, or when its module/class fails to import.
    Adapters flagged ``needs_base_url`` (the vLLM one) are still listed here --
    server reachability is checked at ``setup`` time, not discovery -- but the
    reason is annotated so callers know a ``--base-url`` is required. Imports are
    lazy, so this is safe to call with nothing installed.
    """
    discovered: list[type[PatchSweepEngine]] = []
    for entry in PATCH_SWEEP_REGISTRY:
        if filter_names is not None and entry.name not in filter_names:
            continue

        if not is_package_available(entry.required_package):
            print(
                f"  {entry.name}: SKIPPED ({entry.required_package} not installed)"
            )
            continue

        try:
            module = importlib.import_module(entry.module_path)
            cls = getattr(module, entry.class_name)
        except (ImportError, AttributeError) as exc:
            print(f"  {entry.name}: SKIPPED ({exc})")
            continue

        discovered.append(cls)
        suffix = " (requires a reachable --base-url server)" if entry.needs_base_url else ""
        print(f"  {entry.name}: available{suffix}")

    return discovered


def get_patch_sweep_engine(name: str) -> type[PatchSweepEngine]:
    """Import and return the patch-sweep adapter class named ``name``."""
    for entry in PATCH_SWEEP_REGISTRY:
        if entry.name == name:
            module = importlib.import_module(entry.module_path)
            return getattr(module, entry.class_name)
    known = ", ".join(e.name for e in PATCH_SWEEP_REGISTRY)
    raise PatchSweepError(f"unknown patch-sweep engine {name!r}; known: {known}")
