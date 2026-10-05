"""Concurrency-mode dispatch and startup-failure behaviour of the baked serving handler.

The engine handlers start ONE shared OpenAI server per execution environment. With
AWS_LAMBDA_MAX_CONCURRENCY set the start is deferred to @register_pre_fork so it runs
once in the parent before workers fork; without it the start happens at module level.
A module-level start under multi-concurrency would give N servers racing for the port
and N model copies in VRAM.

Imports happen with subprocess, requests and register_pre_fork patched, so these tests
are CPU-only and never launch an engine.
"""

import contextlib
import importlib
import importlib.util
import os
import sys
from unittest import mock

import pytest

HANDLER = "/var/task/handler.py"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(HANDLER), reason="image ships no baked serving handler"
)


@pytest.fixture
def load(monkeypatch):
    with contextlib.ExitStack() as stack:

        def _load(max_concurrency=None, poll=None, returncode=0, health=200, timeout="2"):
            if max_concurrency is None:
                monkeypatch.delenv("AWS_LAMBDA_MAX_CONCURRENCY", raising=False)
            else:
                monkeypatch.setenv("AWS_LAMBDA_MAX_CONCURRENCY", max_concurrency)
            monkeypatch.setenv("VLLM_SERVER_TIMEOUT", timeout)
            monkeypatch.setenv("SGLANG_SERVER_TIMEOUT", timeout)

            hooks = importlib.import_module("awslambdaric.lambda_concurrency_hooks")
            popen = stack.enter_context(mock.patch("subprocess.Popen"))
            get = stack.enter_context(mock.patch("requests.get"))
            register = stack.enter_context(mock.patch.object(hooks, "register_pre_fork"))
            popen.return_value.poll.return_value = poll
            popen.return_value.returncode = returncode
            get.return_value.status_code = health

            spec = importlib.util.spec_from_file_location("baked_handler", HANDLER)
            module = importlib.util.module_from_spec(spec)
            sys.modules["baked_handler"] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop("baked_handler", None)
            return module, popen, register

        yield _load


def test_multi_concurrency_does_not_start_server_at_import(load):
    _, popen, register = load(max_concurrency="10")
    register.assert_called_once()
    popen.assert_not_called()


def test_registered_pre_fork_hook_starts_the_server(load):
    _, popen, register = load(max_concurrency="10")
    register.call_args[0][0]()
    popen.assert_called_once()


def test_on_demand_starts_server_at_import(load):
    _, popen, register = load()
    popen.assert_called_once()
    register.assert_not_called()


def test_server_is_launched_as_a_shared_localhost_http_server(load):
    _, popen, _ = load()
    cmd = popen.call_args[0][0]
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--port" in cmd


def test_server_exiting_during_startup_raises_with_its_exit_code(load):
    _, _, register = load(max_concurrency="10", poll=1, returncode=1, health=503)
    with pytest.raises(RuntimeError, match=r"exited with code 1"):
        register.call_args[0][0]()


def test_live_but_unready_server_times_out(load):
    _, _, register = load(max_concurrency="10", poll=None, health=503)
    with pytest.raises(RuntimeError, match=r"did not become ready"):
        register.call_args[0][0]()


def test_healthy_server_completes_startup(load):
    _, popen, register = load(max_concurrency="10", poll=None, health=200)
    register.call_args[0][0]()
    popen.assert_called_once()


def test_handler_accepts_both_bytes_and_dict_payloads(load):
    module, _, _ = load()
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {"ok": True}
        assert module.handler({"prompt": "hi"}, None) == {"ok": True}
        assert module.handler(b'{"prompt": "hi"}', None) == {"ok": True}
    assert post.call_count == 2


def test_payload_shape_selects_the_openai_route(load):
    module, _, _ = load()
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {}
        module.handler({"messages": [{"role": "user", "content": "hi"}]}, None)
        assert post.call_args[0][0].endswith("/v1/chat/completions")
        module.handler({"prompt": "hi"}, None)
        assert post.call_args[0][0].endswith("/v1/completions")
