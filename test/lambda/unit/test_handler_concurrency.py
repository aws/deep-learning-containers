"""Single-server election and startup-failure behaviour of the baked serving handler.

Every forked worker runs the handler module, so `execute_once` elects the one worker
that launches the engine; without it each worker would start its own server → N servers
racing for the port and N model copies in VRAM.

Each import below stands for one worker. Imports happen with subprocess and requests
patched, so these tests are CPU-only and never launch an engine.
"""

import contextlib
import importlib.util
import os
import sys
from unittest import mock

import pytest

TASK_ROOT = "/var/task"
HANDLER = f"{TASK_ROOT}/handler.py"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(HANDLER), reason="image ships no baked serving handler"
)

sys.path.insert(0, TASK_ROOT)


def _clear_state():
    for path in (
        "/tmp/vllm_server_start.lock",
        "/tmp/vllm_server_start.result",
        "/tmp/sglang_server_start.lock",
        "/tmp/sglang_server_start.result",
    ):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


@pytest.fixture
def load(monkeypatch):
    """Import the handler as a worker would, returning (module, Popen mock)."""
    _clear_state()
    with contextlib.ExitStack() as stack:

        def _load(poll=None, returncode=0, health=200, timeout="2"):
            monkeypatch.setenv("VLLM_SERVER_TIMEOUT", timeout)
            monkeypatch.setenv("SGLANG_SERVER_TIMEOUT", timeout)

            popen = stack.enter_context(mock.patch("subprocess.Popen"))
            get = stack.enter_context(mock.patch("requests.get"))
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
            return module, popen

        yield _load
    _clear_state()


def test_first_worker_launches_the_server(load):
    _, popen = load()
    popen.assert_called_once()


def test_later_workers_do_not_launch_a_second_server(load):
    load()
    _, popen = load()
    popen.assert_not_called()


def test_server_is_launched_as_a_shared_localhost_http_server(load):
    _, popen = load()
    cmd = popen.call_args[0][0]
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert "--port" in cmd


def test_server_exiting_during_startup_raises_with_its_exit_code(load):
    with pytest.raises(RuntimeError, match=r"exited with code 1"):
        load(poll=1, returncode=1, health=503)


def test_live_but_unready_server_times_out(load):
    with pytest.raises(RuntimeError, match=r"did not become ready"):
        load(health=503)


def test_waiting_worker_times_out_when_the_server_never_becomes_ready(load):
    load()
    with pytest.raises(RuntimeError, match=r"did not become ready"):
        load(health=503)


def test_healthy_server_completes_startup(load):
    _, popen = load(health=200)
    popen.assert_called_once()


def test_handler_accepts_both_bytes_and_dict_payloads(load):
    module, _ = load()
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {"ok": True}
        assert module.handler({"prompt": "hi"}, None) == {"ok": True}
        assert module.handler(b'{"prompt": "hi"}', None) == {"ok": True}
    assert post.call_count == 2


def test_payload_shape_selects_the_openai_route(load):
    module, _ = load()
    with mock.patch("requests.post") as post:
        post.return_value.json.return_value = {}
        module.handler({"messages": [{"role": "user", "content": "hi"}]}, None)
        assert post.call_args[0][0].endswith("/v1/chat/completions")
        module.handler({"prompt": "hi"}, None)
        assert post.call_args[0][0].endswith("/v1/completions")
