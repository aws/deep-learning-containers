"""The baked serving handler is a thin proxy to the engine's OpenAI server.

The engine extension owns the server, so importing the handler must not launch one: each
forked RIC worker imports it, and a launch there would put N model copies in VRAM.
These tests are CPU-only and patch requests, so no engine runs.
"""

import importlib.util
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

HANDLER = "/var/task/handler.py"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(HANDLER), reason="image ships no baked serving handler"
)


@pytest.fixture
def module():
    with mock.patch("subprocess.Popen") as popen:
        spec = importlib.util.spec_from_file_location("baked_handler", HANDLER)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["baked_handler"] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop("baked_handler", None)
    popen.assert_not_called()
    return mod


def _context(remaining_ms):
    return SimpleNamespace(get_remaining_time_in_millis=lambda: remaining_ms)


def test_handler_accepts_both_bytes_and_dict_payloads(module):
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {"ok": True}
        assert module.handler({"prompt": "hi"}, None) == {"ok": True}
        assert module.handler(b'{"prompt": "hi"}', None) == {"ok": True}
    assert post.call_count == 2


def test_payload_shape_selects_the_openai_route(module):
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {}
        module.handler({"messages": [{"role": "user", "content": "hi"}]}, None)
        assert post.call_args[0][0].endswith("/v1/chat/completions")
        module.handler({"prompt": "hi"}, None)
        assert post.call_args[0][0].endswith("/v1/completions")


def test_requests_go_to_the_local_server(module):
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {}
        module.handler({"prompt": "hi"}, None)
    assert post.call_args[0][0].startswith("http://127.0.0.1:")


def test_request_timeout_follows_the_invocation_deadline(module):
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {}
        module.handler({"prompt": "hi"}, _context(30_000))
        assert post.call_args.kwargs["timeout"] == pytest.approx(29)
        module.handler({"prompt": "hi"}, _context(200))
        assert post.call_args.kwargs["timeout"] == pytest.approx(0.1)


def test_rejected_request_surfaces_the_server_status_and_body(module):
    with mock.patch("requests.post") as post:
        post.return_value.ok = False
        post.return_value.status_code = 400
        post.return_value.text = '{"message": "When using `tool_choice`, `tools` must be set."}'
        with pytest.raises(RuntimeError, match=r"400: .*`tools` must be set"):
            module.handler({"messages": [], "tool_choice": "auto"}, None)


def test_unreachable_server_raises_a_clear_error(module):
    with mock.patch("requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match=r"server is not reachable at http://127\.0\.0\.1:"):
            module.handler({"prompt": "hi"}, None)


def test_request_past_the_deadline_raises_a_clear_error(module):
    with mock.patch("requests.post", side_effect=requests.ReadTimeout("read timed out")):
        with pytest.raises(RuntimeError, match=r"did not answer before the invocation deadline"):
            module.handler({"prompt": "hi"}, _context(30_000))
