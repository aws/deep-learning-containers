"""Engine extension lifecycle against a fake Lambda Extensions API and a fake engine.

The extension (/opt/engine-extension/engine_extension.py) owns the vLLM/SGLang server.
Each test runs it as a subprocess, as Lambda does, with the engine command swapped for a
tiny HTTP server whose /health status the test controls. CPU-only: no engine, no GPU.
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# In the image, or in a repo checkout (test/lambda/unit/ -> scripts/docker/lambda/).
CANDIDATES = [Path("/opt/engine-extension/engine_extension.py")] + [
    parent / "scripts/docker/lambda/engine_extension.py"
    for parent in Path(__file__).resolve().parents
]
EXTENSION = next((p for p in CANDIDATES if p.is_file()), None)

pytestmark = pytest.mark.skipif(EXTENSION is None, reason="image ships no engine extension")

# Stands in for `vllm serve`: answers /health with the status in STATUS_FILE, after an
# optional startup delay, and records its pid so tests can check it was stopped, and the
# environment it was started with (PID_FILE.env). Like the real engines it starts a worker
# process (pid in PID_FILE.worker) that outlives it.
FAKE_ENGINE = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, status_file, pid_file, delay = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    # Only the keys the tests check: CI's environment may hold real credentials.
    keys = [k for k in os.environ if k.startswith(("VLLM_", "SGLANG_")) or k in ("MODEL_ID", "HF_TOKEN")]
    seen = {k: os.environ[k] for k in keys}
    seen["PATH_SET"] = "PATH" in os.environ
    open(pid_file + ".env", "w").write(json.dumps(seen))
    open(pid_file, "w").write(str(os.getpid()))
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    open(pid_file + ".worker", "w").write(str(worker.pid))
    time.sleep(delay)
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(int(open(status_file).read().strip() or 200))
            self.end_headers()
        def log_message(self, *args):
            pass
    HTTPServer(("127.0.0.1", int(port)), H).serve_forever()
    """
)

# Runs the extension's main() with a "fake" engine registered, as /opt/extensions would.
RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import engine_extension as ext
    cmd = json.loads(sys.argv[2])
    ext.ENGINES["fake"] = (lambda env: cmd, "FAKE", 10)
    sys.exit(ext.main(["engine_extension.py", "fake", "fake-engine"]))
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeExtensionsApi:
    """Records Extensions API calls; event/next blocks until shutdown() is called."""

    def __init__(self):
        self.calls = []
        self.next_called = threading.Event()
        self._shutdown = threading.Event()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                path = self.path.removeprefix("/2020-01-01/extension/")
                api.calls.append(
                    {
                        "method": self.command,
                        "path": path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                return path

            def do_POST(self):
                path = self._record()
                if path == "register":
                    self.send_response(200)
                    self.send_header("Lambda-Extension-Identifier", "ext-id")
                    self.end_headers()
                    self.wfile.write(b'{"functionName": "f"}')
                else:
                    self.send_response(202)
                    self.end_headers()

            def do_GET(self):
                self._record()
                api.next_called.set()
                api._shutdown.wait()
                payload = json.dumps(
                    {"eventType": "SHUTDOWN", "shutdownReason": "SPINDOWN"}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.address = f"127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def shutdown(self):
        self._shutdown.set()

    def paths(self):
        return [c["path"] for c in self.calls]

    def call(self, path):
        return next(c for c in self.calls if c["path"] == path)

    def close(self):
        self._shutdown.set()
        self.server.shutdown()


class Setup:
    def __init__(self, tmp_path, api):
        self.api = api
        self.port = _free_port()
        self.status_file = tmp_path / "status"
        self.pid_file = tmp_path / "engine.pid"
        self.engine_script = tmp_path / "fake_engine.py"
        self.engine_script.write_text(FAKE_ENGINE)
        self.set_health(200)
        self.proc = None

    def set_health(self, status):
        self.status_file.write_text(str(status))

    def start(self, delay=0.0, engine_cmd=None, **env):
        cmd = engine_cmd or [
            sys.executable,
            str(self.engine_script),
            str(self.port),
            str(self.status_file),
            str(self.pid_file),
            str(delay),
        ]
        full_env = {
            **os.environ,
            "AWS_LAMBDA_RUNTIME_API": self.api.address,
            "FAKE_SERVER_PORT": str(self.port),
            "FAKE_SERVER_TIMEOUT": "10",
            "ENGINE_HEALTH_INTERVAL": "1",
            "ENGINE_HEALTH_TIMEOUT": "1",
            "ENGINE_HEALTH_FAILURES": "2",
            **env,
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-c", RUNNER, str(EXTENSION.parent), json.dumps(cmd)], env=full_env
        )
        return self.proc

    def wait(self, timeout=30):
        return self.proc.wait(timeout=timeout)

    def engine_pid(self):
        return int(self.pid_file.read_text())

    def worker_pid(self):
        return int(Path(f"{self.pid_file}.worker").read_text())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for pid_file in (self.pid_file, Path(f"{self.pid_file}.worker")):
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it's gone for our purposes.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _wait_dead(pid, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def setup(tmp_path):
    api = FakeExtensionsApi()
    s = Setup(tmp_path, api)
    yield s
    s.stop()
    api.close()


def test_registers_for_shutdown_only_under_its_file_name(setup):
    setup.start()
    assert setup.api.next_called.wait(30)
    register = setup.api.call("register")
    # Lambda Managed Instances rejects INVOKE registrations.
    assert register["body"] == {"events": ["SHUTDOWN"]}
    assert register["headers"]["Lambda-Extension-Name"] == "fake-engine"
    assert setup.api.call("event/next")["headers"]["Lambda-Extension-Identifier"] == "ext-id"


def test_init_completes_only_after_the_engine_is_healthy(setup):
    setup.start(delay=2)
    started = time.monotonic()
    assert setup.api.next_called.wait(30)
    assert time.monotonic() - started >= 2
    assert "init/error" not in setup.api.paths()


def test_shutdown_stops_the_engine_and_exits_cleanly(setup):
    setup.start()
    assert setup.api.next_called.wait(30)
    pid = setup.engine_pid()
    setup.api.shutdown()
    assert setup.wait() == 0
    assert _wait_dead(pid)
    assert _wait_dead(setup.worker_pid())
    # A deliberate stop is not an engine failure.
    assert "exit/error" not in setup.api.paths()


def test_engine_exiting_during_startup_reports_an_init_error(setup):
    setup.start(engine_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"])
    assert setup.wait() == 1
    error = setup.api.call("init/error")
    assert error["headers"]["Lambda-Extension-Function-Error-Type"] == "Extension.EngineInitFailed"
    assert "exited with code 3" in error["body"]["errorMessage"]
    assert "event/next" not in setup.api.paths()


def test_engine_never_healthy_reports_an_init_error_and_is_stopped(setup):
    setup.set_health(503)
    setup.start(FAKE_SERVER_TIMEOUT="3")
    assert setup.wait() == 1
    assert (
        "did not become healthy within 3s" in setup.api.call("init/error")["body"]["errorMessage"]
    )
    assert _wait_dead(setup.engine_pid())


@pytest.mark.parametrize(
    "env, message",
    [
        ({"FAKE_SERVER_TIMEOUT": "ten"}, "invalid literal for int()"),
        ({"ENGINE_HEALTH_INTERVAL": "5s"}, "invalid literal for int()"),
        ({"ENGINE_HEALTH_FAILURES": "0"}, "ENGINE_HEALTH_FAILURES must be a positive integer"),
    ],
)
def test_invalid_setting_reports_an_init_error_naming_it(setup, env, message):
    setup.start(**env)
    assert setup.wait() == 1
    error = setup.api.call("init/error")
    assert error["headers"]["Lambda-Extension-Function-Error-Type"] == "Extension.EngineInitFailed"
    assert message in error["body"]["errorMessage"]


def test_engine_dying_after_init_reports_an_exit_error(setup):
    setup.start()
    assert setup.api.next_called.wait(30)
    os.kill(setup.engine_pid(), signal.SIGKILL)
    assert setup.wait(timeout=15) == 1
    error = setup.api.call("exit/error")
    assert error["headers"]["Lambda-Extension-Function-Error-Type"] == "Extension.EngineUnhealthy"
    assert "engine exited with code" in error["body"]["errorMessage"]
    # The engine's workers would otherwise keep holding GPU memory.
    assert _wait_dead(setup.worker_pid())


def test_unhealthy_engine_after_init_reports_an_exit_error_and_is_stopped(setup):
    setup.start()
    assert setup.api.next_called.wait(30)
    setup.set_health(503)
    assert setup.wait(timeout=20) == 1
    assert "2 consecutive health probes" in setup.api.call("exit/error")["body"]["errorMessage"]
    assert _wait_dead(setup.engine_pid())


def test_a_single_failed_probe_does_not_recycle_the_environment(setup):
    setup.start(ENGINE_HEALTH_FAILURES="3")
    assert setup.api.next_called.wait(30)
    setup.set_health(503)
    time.sleep(1.5)
    setup.set_health(200)
    time.sleep(4)
    assert setup.proc.poll() is None
    assert "exit/error" not in setup.api.paths()


def test_disabled_extension_registers_without_starting_an_engine(setup):
    setup.start(ENGINE_EXTENSION="off")
    assert setup.api.next_called.wait(30)
    assert not setup.pid_file.exists()
    setup.api.shutdown()
    assert setup.wait() == 0


def test_engine_does_not_see_the_image_settings(setup):
    # vLLM warns about every VLLM_* variable it doesn't know, and fails to start on them
    # with --fail-on-environ-validation; the image's own settings are for the extension.
    setup.start(
        VLLM_MAX_MODEL_LEN="4096",
        VLLM_SERVER_ARGS="--enforce-eager",
        SGLANG_MEM_FRACTION="0.5",
        VLLM_LOGGING_LEVEL="DEBUG",
        HF_TOKEN="hf_test",
        MODEL_ID="s3://bucket/model/",
    )
    assert setup.api.next_called.wait(30)
    engine_env = json.loads(Path(f"{setup.pid_file}.env").read_text())
    for name in ("VLLM_MAX_MODEL_LEN", "VLLM_SERVER_ARGS", "SGLANG_MEM_FRACTION"):
        assert name not in engine_env, name
    # Everything else still reaches the engine, including genuine vLLM settings.
    assert engine_env["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert engine_env["HF_TOKEN"] == "hf_test"
    assert engine_env["MODEL_ID"] == "s3://bucket/model/"
    assert engine_env["PATH_SET"]


@pytest.fixture
def ext(monkeypatch):
    monkeypatch.syspath_prepend(str(EXTENSION.parent))
    import engine_extension

    monkeypatch.setattr(engine_extension, "gpu_count", lambda env=None: 1)
    return engine_extension


def test_vllm_command_binds_localhost_and_takes_extra_args(ext):
    cmd = ext.vllm_command(
        {
            "MODEL_ID": "s3://bucket/model/",
            "VLLM_SERVER_PORT": "8123",
            "VLLM_LOAD_FORMAT": "runai_streamer",
            "SERVED_MODEL_NAME": "m",
            "VLLM_SERVER_ARGS": "--enable-prefix-caching --max-num-seqs '64'",
        }
    )
    assert cmd[0].endswith("/vllm") and cmd[1:3] == ["serve", "s3://bucket/model/"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert cmd[cmd.index("--load-format") + 1] == "runai_streamer"
    assert cmd[cmd.index("--served-model-name") + 1] == "m"
    assert cmd[-3:] == ["--enable-prefix-caching", "--max-num-seqs", "64"]


def test_vllm_extra_args_with_an_unbalanced_quote_raise(ext):
    # run() turns this into an init error; the shlex message names the problem.
    with pytest.raises(ValueError, match="quotation"):
        ext.vllm_command({"VLLM_SERVER_ARGS": "--chat-template '{{x}}"})


@pytest.mark.parametrize("visible, expected", [("0", 1), ("0,1,3", 3), ("", 1)])
def test_default_parallel_size_follows_cuda_visible_devices(visible, expected, monkeypatch):
    monkeypatch.syspath_prepend(str(EXTENSION.parent))
    import engine_extension

    assert engine_extension.gpu_count({"CUDA_VISIBLE_DEVICES": visible}) == expected


def test_sglang_command_binds_localhost_and_takes_extra_args(ext):
    cmd = ext.sglang_command({"SGLANG_TP_SIZE": "2", "SGLANG_SERVER_ARGS": "--enable-metrics"})
    assert cmd[1:3] == ["-m", "sglang.launch_server"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--tp-size") + 1] == "2"
    assert cmd[cmd.index("--model-path") + 1] == ext.DEFAULT_MODEL
    assert cmd[-1] == "--enable-metrics"


def test_every_engine_setting_the_extension_reads_is_kept_from_the_engine(ext):
    # Guards against a new VLLM_* / SGLANG_* setting being read without being listed: every
    # such name in the extension's source (any way it is read, including f"{prefix}_..."
    # names in run()) must be in IMAGE_SETTINGS.
    source = Path(ext.__file__).read_text()
    names = set(re.findall(r"\b(?:VLLM|SGLANG)_[A-Z0-9_]*[A-Z0-9]\b", source))
    for suffix in re.findall(r"\{prefix\}(_[A-Z0-9_]*[A-Z0-9])", source):
        names |= {f"{prefix}{suffix}" for _, prefix, _ in ext.ENGINES.values()}
    assert names, "found no settings in the extension's source"
    assert names <= ext.IMAGE_SETTINGS, names - ext.IMAGE_SETTINGS
