"""Server-error surfacing for the vLLM patch-sweep client.

The client posts to a server it does not launch; a misconfigured server (most
commonly one missing ``--patch-source-cache-bytes``) rejects the sweep with an
informative JSON ``{"error": ...}`` body. These tests pin the pure extraction
helper — no httpx, no live server.
"""

from __future__ import annotations

from steering_bench.external.vllm_patch_sweep import (
    PatchSweepServerError,
    _server_error_detail,
)


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def test_error_detail_includes_server_message_and_flag_hint() -> None:
    resp = _FakeResp(
        400,
        {
            "error": "patch source not found: run='x' "
            "site=(layer=0, hook=post_block, position=0)."
        },
    )
    detail = _server_error_detail(resp)
    assert "patch source not found" in detail
    assert "HTTP 400" in detail
    # the launch-flag hint is appended for the missing-source condition
    assert "patch-source-cache-bytes" in detail
    assert "--capture-consumers patch_source" in detail


def test_error_detail_no_hint_for_unrelated_error() -> None:
    resp = _FakeResp(400, {"error": "answer_token is required"})
    detail = _server_error_detail(resp)
    assert "answer_token is required" in detail
    assert "patch-source-cache-bytes" not in detail


def test_error_detail_falls_back_to_text_when_no_json() -> None:
    resp = _FakeResp(500, payload=None, text="Internal Server Error")
    detail = _server_error_detail(resp)
    assert "Internal Server Error" in detail
    assert "HTTP 500" in detail


def test_enable_patching_message_gets_hint() -> None:
    resp = _FakeResp(400, {"error": "server was not started with --enable-patching"})
    detail = _server_error_detail(resp)
    assert "patch-source-cache-bytes" in detail


def test_server_error_is_runtimeerror() -> None:
    assert issubclass(PatchSweepServerError, RuntimeError)
