"""Base-url normalization for the vLLM patch-sweep HTTP driver.

Finding #2 (e2e validation): the serving path auto-appends ``/v1`` while
patch-sweep used to require the caller to pass a ``/v1``-suffixed URL. The
driver now normalizes either form to the same ``/v1`` root, so a bare host and
a ``/v1``-suffixed host resolve to the same POST URL and health root. No httpx
import / live server required.
"""

from __future__ import annotations

from steering_bench.external.vllm_patch_sweep import normalize_base_url

BARE = "http://host:8000"
V1 = "http://host:8000/v1"
TRAILING = "http://host:8000/"


def _post_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/patch_sweep"


def _health_root(base_url: str) -> str:
    return normalize_base_url(base_url).removesuffix("/v1")


def test_normalize_appends_v1_to_bare_host() -> None:
    assert normalize_base_url(BARE) == V1


def test_normalize_is_idempotent_on_v1_suffix() -> None:
    assert normalize_base_url(V1) == V1


def test_normalize_strips_trailing_slash() -> None:
    assert normalize_base_url(TRAILING) == V1
    assert normalize_base_url(V1 + "/") == V1


def test_bare_and_v1_resolve_to_same_post_url() -> None:
    assert _post_url(BARE) == _post_url(V1) == "http://host:8000/v1/patch_sweep"


def test_bare_and_v1_resolve_to_same_health_root() -> None:
    assert _health_root(BARE) == _health_root(V1) == "http://host:8000"
