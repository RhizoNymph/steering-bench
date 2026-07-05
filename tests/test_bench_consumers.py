"""CPU-side construction + emission tests for the dynamic-steering bench
consumers. No CUDA: we construct each consumer with a mock ``vllm_config``,
call ``global_capture_spec``, and drive ``on_step`` with fake step views to
assert action shapes / ordering / dedup / live-set pruning.

Run from the vLLM fork dir so cwd-shadowing loads the branch under test::

    cd /home/nymph/Code/vllm/dynamic-steering && \
    PYTHONPATH=/home/nymph/Code/steering-bench/src \
    /home/nymph/Code/vllm/integration-v2/.venv/bin/python -m pytest \
      /home/nymph/Code/steering-bench/tests/test_bench_consumers.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from steering_bench.capture_consumers import bench_consumers as bc
from vllm.model_executor.layers.steering import (
    VALID_HOOK_POINT_NAMES,
    SteeringHookPoint,
)
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringVectorUpdate,
)

HIDDEN = 2560


class _MockModelConfig:
    def get_hidden_size(self) -> int:
        return HIDDEN


class _MockVllmConfig:
    model_config = _MockModelConfig()


@dataclass
class _FakeReq:
    req_id: str
    phase: str


class _FakeView:
    def __init__(self, reqs: list[_FakeReq]) -> None:
        self.requests = reqs


def _cfg() -> _MockVllmConfig:
    return _MockVllmConfig()


@pytest.fixture(autouse=True)
def _clean_registry():
    bc.reset_live_consumers()
    yield
    bc.reset_live_consumers()


def test_default_hook_is_valid_enum_member():
    """The renamed default hook must be a live enum member (post_mlp is gone)."""
    assert bc._HOOK == SteeringHookPoint.POST_BLOCK.value
    assert bc._HOOK in VALID_HOOK_POINT_NAMES


def test_multi_site_vectors_distinct_equal_norm():
    c = bc.BenchSteerSync(
        _cfg(),
        {"layers": [3, 17], "hooks": ["pre_attn", "post_block"], "norm": 8.0},
    )
    sites = [(h, layer) for h, layer in c._sites]
    assert set(sites) == {
        ("pre_attn", 3), ("pre_attn", 17),
        ("post_block", 3), ("post_block", 17),
    }
    vecs = [c._vectors[h][layer] for h, layer in sites]
    for v in vecs:
        assert v.shape == (HIDDEN,)
        assert np.isclose(np.linalg.norm(v), 8.0, rtol=1e-4)
    # distinct per site
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            assert not np.allclose(vecs[i], vecs[j])


def test_back_compat_single_layer_hook():
    c = bc.BenchSteerSync(_cfg(), {"layer": 30, "hook": "post_attn", "norm": 4.0})
    assert c._sites == [("post_attn", 30)]
    assert np.isclose(np.linalg.norm(c._vectors["post_attn"][30]), 4.0, rtol=1e-4)


def test_capture_spec_covers_all_sites_and_monitor():
    c = bc.BenchSteerRowmon(
        _cfg(),
        {
            "layers": [5, 10], "hooks": ["post_block"],
            "monitor_layer": 20, "monitor_hook": "pre_attn", "norm": 8.0,
        },
    )
    spec = c.global_capture_spec()
    assert set(spec.hooks["post_block"]) == {5, 10}
    assert 20 in spec.hooks["pre_attn"]
    assert spec.positions == "all_generated"


def test_steer_sync_emits_single_global_tier_update_once():
    c = bc.BenchSteerSync(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    view = _FakeView([_FakeReq("r0", "decode")])
    out = c.on_step(view)
    assert out is not None and len(out) == 1
    assert isinstance(out[0], SteeringVectorUpdate)
    assert out[0].phase == "decode"
    assert list(out[0].vectors["post_block"].keys()) == [17]
    # idempotent: nothing more after the first emit
    assert c.on_step(view) is None


def test_steer_dynamic_installs_tier_then_global_monitor():
    c = bc.BenchSteerDynamic(
        _cfg(), {"layers": [17], "hooks": ["post_block"], "monitor_layer": 17}
    )
    out = c.on_step(_FakeView([_FakeReq("r0", "decode")]))
    assert [type(a) for a in out] == [SteeringVectorUpdate, SteeringMonitorUpdate]
    mon = out[1]
    assert mon.req_id is None  # GLOBAL monitor
    assert mon.threshold < 0 and mon.sharpness > 0


def test_override_one_per_request_deduped_and_pruned():
    c = bc.BenchSteerOverride(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    v1 = _FakeView([_FakeReq("a", "decode"), _FakeReq("b", "decode")])
    out = c.on_step(v1)
    assert out is not None
    assert all(isinstance(a, RequestSteeringOverride) for a in out)
    assert {a.req_id for a in out} == {"a", "b"}
    assert c._SOURCE == "bench_steer_override"
    # same requests next step -> already emitted -> nothing new
    assert c.on_step(v1) is None
    # a finishes, c appears -> only c gets a fresh override; a is pruned
    v2 = _FakeView([_FakeReq("b", "decode"), _FakeReq("c", "decode")])
    out2 = c.on_step(v2)
    assert {a.req_id for a in out2} == {"c"}
    assert "a" not in c._emitted


def test_override_skips_prefill_phase():
    c = bc.BenchSteerOverride(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    out = c.on_step(_FakeView([_FakeReq("a", "prefill")]))
    assert out is None


def test_rowmon_emits_monitor_after_override_same_list():
    c = bc.BenchSteerRowmon(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    out = c.on_step(_FakeView([_FakeReq("a", "decode"), _FakeReq("b", "decode")]))
    assert out is not None and len(out) == 4
    # ordering: for each request the override precedes its per-row monitor
    assert isinstance(out[0], RequestSteeringOverride) and out[0].req_id == "a"
    assert isinstance(out[1], SteeringMonitorUpdate) and out[1].req_id == "a"
    assert isinstance(out[2], RequestSteeringOverride) and out[2].req_id == "b"
    assert isinstance(out[3], SteeringMonitorUpdate) and out[3].req_id == "b"
    for a in (out[1], out[3]):
        assert a.threshold < 0 and a.sharpness > 0  # saturated -> gate ~1
    assert c._SOURCE == "bench_steer_rowmon"


def test_every_consumer_class_exposes_declared_graphsafe_keys():
    """The capture registry calls this on the CLASS at config-build time
    (outside a try/except), so a sync-only consumer missing it crashes
    registration — a path the instance-level tests never exercise."""
    for cls in (
        bc.BenchCaptureAsync, bc.BenchCaptureSync,
        bc.BenchSteerAsync, bc.BenchSteerSync, bc.BenchSteerDynamic,
        bc.BenchSteerOverride, bc.BenchSteerRowmon, bc.BenchSteerPerRequest,
    ):
        assert cls.declared_graphsafe_keys({"layer": 17}) == []


def test_per_request_distinct_vectors_no_prune():
    c = bc.BenchSteerPerRequest(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    out = c.on_step(_FakeView([_FakeReq("a", "decode"), _FakeReq("b", "decode")]))
    assert {a.req_id for a in out} == {"a", "b"}
    va = out[0].vectors["post_block"][17]
    vb = out[1].vectors["post_block"][17]
    assert not np.allclose(va, vb)  # distinct per request
    assert out[0].source == "bench_steer_per_request"
    assert c.on_step(_FakeView([_FakeReq("a", "decode")])) is None  # a already seen


def test_live_consumer_registry_tracks_instances():
    bc.reset_live_consumers()
    a = bc.BenchCaptureSync(_cfg(), {"layer": 17})
    b = bc.BenchSteerOverride(_cfg(), {"layers": [17], "hooks": ["post_block"]})
    live = bc.iter_live_consumers()
    assert a in live and b in live
    a.on_step(_FakeView([_FakeReq("x", "decode")]))
    assert a._steps == 1
