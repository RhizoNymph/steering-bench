"""Engine registry + capability-aware discovery.

Generalizes the ``discover_libraries`` / ``LIBRARY_REGISTRY`` pattern from
``scripts/bench_external.py``: engines are declared as data (name, required
package, import path, capabilities) and imported lazily, so declaring an engine
costs nothing until it is actually used.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

from steering_bench.engine.base import Capabilities, SteeringEngine


@dataclass(frozen=True)
class EngineEntry:
    """A registered engine adapter, described without importing it."""

    name: str
    required_package: str | None
    module_path: str
    class_name: str
    capabilities: Capabilities


ENGINE_REGISTRY: list[EngineEntry] = [
    EngineEntry(
        name="vllm",
        required_package="vllm",
        module_path="steering_bench.engine.engines.vllm",
        class_name="VllmSteeringEngine",
        capabilities=Capabilities(
            batching=True,
            named_modules=True,
            multi_layer=True,
            multi_hook=True,
            capture=True,
            prefix_cache=True,
            config_capacity=True,
        ),
    ),
    EngineEntry(
        name="transformerlens",
        required_package="transformer_lens",
        module_path="steering_bench.engine.engines.transformerlens",
        class_name="TransformerLensSteeringEngine",
        capabilities=Capabilities(
            batching=False,
            named_modules=False,
            multi_layer=False,
            multi_hook=False,
            capture=False,
        ),
    ),
    EngineEntry(
        name="hf_baseline",
        required_package="transformers",
        module_path="steering_bench.engine.engines.hf",
        class_name="HFSteeringEngine",
        capabilities=Capabilities(
            batching=True,
            named_modules=False,
            multi_layer=False,
            multi_hook=False,
            capture=False,
        ),
    ),
    EngineEntry(
        name="nnsight",
        required_package="nnsight",
        module_path="steering_bench.engine.engines.nnsight",
        class_name="NnsightSteeringEngine",
        capabilities=Capabilities(
            batching=True,
            named_modules=False,
            multi_layer=False,
            multi_hook=False,
            capture=False,
        ),
    ),
    EngineEntry(
        name="repeng",
        required_package="repeng",
        module_path="steering_bench.engine.engines.repeng",
        class_name="RepengSteeringEngine",
        capabilities=Capabilities(
            batching=False,
            named_modules=False,
            multi_layer=True,
            multi_hook=False,
            capture=False,
        ),
    ),
    EngineEntry(
        name="pyvene",
        required_package="pyvene",
        module_path="steering_bench.engine.engines.pyvene",
        class_name="PyveneSteeringEngine",
        capabilities=Capabilities(
            batching=False,
            named_modules=False,
            multi_layer=False,
            multi_hook=False,
            capture=False,
        ),
    ),
]


def is_package_available(package: str | None) -> bool:
    """Whether ``package`` is importable. ``None`` means no dependency."""
    if package is None:
        return True
    return importlib.util.find_spec(package) is not None


def discover(
    filter_names: list[str] | None = None,
    required: Capabilities | None = None,
) -> list[type[SteeringEngine]]:
    """Return engine classes that are usable in this environment.

    An engine is skipped (with a printed reason) when:
      * it is not in ``filter_names`` (when a filter is given),
      * its capabilities do not satisfy ``required``,
      * its required package is not installed,
      * or its adapter module/class fails to import.

    Imports are lazy: adapter modules must not import their heavy backend at
    module scope, so this is safe to call with nothing installed.
    """
    discovered: list[type[SteeringEngine]] = []
    for entry in ENGINE_REGISTRY:
        if filter_names is not None and entry.name not in filter_names:
            continue

        if required is not None and not entry.capabilities.satisfies(required):
            print(
                f"  {entry.name}: SKIPPED (capabilities do not satisfy "
                f"required {required})"
            )
            continue

        if not is_package_available(entry.required_package):
            print(f"  {entry.name}: SKIPPED ({entry.required_package} not installed)")
            continue

        try:
            module = importlib.import_module(entry.module_path)
            cls = getattr(module, entry.class_name)
        except (ImportError, AttributeError) as exc:
            print(f"  {entry.name}: SKIPPED ({exc})")
            continue

        discovered.append(cls)
        print(f"  {entry.name}: available")

    return discovered
