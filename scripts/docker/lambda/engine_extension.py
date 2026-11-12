"""Lambda extension that owns the serving engine (vLLM or SGLang) of a Lambda GPU image.

Installed at /opt/extensions/<engine>-engine (a shell wrapper that runs this file with the
engine name). Lambda starts every executable in /opt/extensions before the runtime, and
the execution environment's Init phase only ends once each extension asks for its next
event, so no invocation reaches the handler before the engine is ready.

Lifecycle:
  Init      Register for SHUTDOWN (the only event Lambda Managed Instances allows), launch
            the engine's OpenAI-compatible server on 127.0.0.1, and wait for /health. On
            failure, report /extension/init/error and exit: Lambda discards the
            execution environment instead of routing traffic to it.
  Serving   A watchdog thread checks that the engine process is alive and that /health
            answers. After ENGINE_HEALTH_FAILURES consecutive failed probes, or as soon as
            the process exits, it reports /extension/exit/error, stops the engine and
            exits, so Lambda replaces the environment.
  Shutdown  Stop the engine (SIGTERM, then SIGKILL within Lambda's 2 s budget) and exit.

The handler (/var/task/handler.py) only proxies requests to the server.

Environment variables (engine-specific ones are listed in each engine's handler.py):
  ENGINE_EXTENSION          "off" registers and idles without starting an engine, for
                            functions that run their own server (default: on)
  ENGINE_HEALTH_INTERVAL    seconds between health probes after Init (default: 10)
  ENGINE_HEALTH_TIMEOUT     seconds per health probe (default: 10 for vLLM, 30 for SGLang,
                            whose /health may run a one-token generation with its own 20 s
                            timeout; it answers within ~1 s while the server is busy)
  ENGINE_HEALTH_FAILURES    consecutive failed probes before giving up (default: 3)
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

EXTENSIONS_API = "2020-01-01/extension"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Lambda gives external extensions 2 s for the whole Shutdown phase: SIGTERM, wait this
# long, then SIGKILL and wait up to 0.5 s more.
STOP_GRACE_SECONDS = 1.0


def log(message):
    print(f"[engine-extension] {message}", file=sys.stderr, flush=True)


def gpu_count(env=os.environ):
    """Visible GPUs, without importing torch or creating a CUDA context in this process."""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return max(1, len([d for d in visible.split(",") if d.strip()]))
    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=30
        )
        count = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        count = 0
    return max(1, count)


def _tp_size(env, name):
    return str(int(env.get(name, "0") or "0") or gpu_count(env))


def vllm_command(env):
    cmd = [
        # The interpreter's own bin/: an extension's PATH is not guaranteed to be the image's.
        os.path.join(os.path.dirname(sys.executable), "vllm"),
        "serve",
        env.get("MODEL_ID", DEFAULT_MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        env.get("VLLM_SERVER_PORT", "8000"),
        "--gpu-memory-utilization",
        env.get("VLLM_GPU_MEM_UTIL", "0.8"),
        "--tensor-parallel-size",
        _tp_size(env, "VLLM_TP_SIZE"),
    ]
    if env.get("VLLM_MAX_MODEL_LEN"):
        cmd += ["--max-model-len", env["VLLM_MAX_MODEL_LEN"]]
    if env.get("VLLM_LOAD_FORMAT"):
        cmd += ["--load-format", env["VLLM_LOAD_FORMAT"]]
    if env.get("SERVED_MODEL_NAME"):
        cmd += ["--served-model-name", env["SERVED_MODEL_NAME"]]
    return cmd + shlex.split(env.get("VLLM_SERVER_ARGS", ""))


def sglang_command(env):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        env.get("MODEL_ID", DEFAULT_MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        env.get("SGLANG_SERVER_PORT", "8000"),
        "--mem-fraction-static",
        env.get("SGLANG_MEM_FRACTION", "0.8"),
        "--attention-backend",
        env.get("SGLANG_ATTENTION_BACKEND", "flashinfer"),
        "--tp-size",
        _tp_size(env, "SGLANG_TP_SIZE"),
    ]
    if env.get("SGLANG_MAX_TOTAL_TOKENS"):
        cmd += ["--max-total-tokens", env["SGLANG_MAX_TOTAL_TOKENS"]]
    if env.get("SGLANG_LOAD_FORMAT"):
        cmd += ["--load-format", env["SGLANG_LOAD_FORMAT"]]
    if env.get("SERVED_MODEL_NAME"):
        cmd += ["--served-model-name", env["SERVED_MODEL_NAME"]]
    return cmd + shlex.split(env.get("SGLANG_SERVER_ARGS", ""))


ENGINES = {
    # engine: (command builder, env prefix for port and startup timeout, default probe timeout)
    "vllm": (vllm_command, "VLLM", 10),
    "sglang": (sglang_command, "SGLANG", 30),
}


class ExtensionsClient:
    """Minimal client for the Lambda Extensions API."""

    def __init__(self, runtime_api, name):
        self.base = f"http://{runtime_api}/{EXTENSIONS_API}"
        self.name = name
        self.id = None

    def _request(self, method, path, body=None, headers=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base}/{path}", data=data, method=method, headers=headers or {}
        )
        if data is not None:
            req.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(req, timeout=timeout)

    def register(self):
        with self._request(
            "POST",
            "register",
            {"events": ["SHUTDOWN"]},
            {"Lambda-Extension-Name": self.name},
            timeout=30,
        ) as resp:
            self.id = resp.headers["Lambda-Extension-Identifier"]

    def next_event(self):
        # No timeout: the call blocks until Lambda has an event, possibly for hours.
        with self._request(
            "GET", "event/next", headers={"Lambda-Extension-Identifier": self.id}
        ) as resp:
            return json.loads(resp.read() or b"{}")

    def report_error(self, phase, error_type, message):
        """Report an init or exit error. Best effort: the extension exits either way."""
        try:
            self._request(
                "POST",
                f"{phase}/error",
                {"errorMessage": message, "errorType": error_type, "stackTrace": []},
                {
                    "Lambda-Extension-Identifier": self.id,
                    "Lambda-Extension-Function-Error-Type": error_type,
                },
                timeout=5,
            ).close()
        except (urllib.error.URLError, OSError) as e:
            log(f"could not report {phase} error: {e}")


def _group_alive(pgid):
    """True while any non-zombie process is in the group.

    Zombies don't count: under the RIE, PID 1 does not reap orphaned workers, so they
    would keep the group "alive" after every process in it has died.
    """
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as f:
                # Fields after the parenthesised command: state, ppid, pgrp, ...
                fields = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False


class Engine:
    """The engine's server process and its health endpoint."""

    def __init__(self, cmd, health_url):
        self.cmd = cmd
        self.health_url = health_url
        self.proc = None
        # Set before a deliberate stop, so the watchdog does not report it as a failure.
        self.stopping = threading.Event()

    def start(self):
        log(f"starting: {shlex.join(self.cmd)}")
        # Own process group: the engines fork workers (vLLM's engine core, SGLang's
        # scheduler and detokenizer) that must stop with the server and free the GPU.
        self.proc = subprocess.Popen(self.cmd, start_new_session=True)

    def exit_code(self):
        return self.proc.poll()

    def healthy(self, timeout):
        try:
            with urllib.request.urlopen(self.health_url, timeout=timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, timeout, probe_timeout):
        """Block until /health answers 200; raise if the server exits or times out."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self.exit_code()
            if code is not None:
                raise RuntimeError(f"engine exited with code {code} during startup")
            if self.healthy(probe_timeout):
                return
            time.sleep(1)
        raise RuntimeError(f"engine did not become healthy within {timeout}s")

    def stop(self, grace=STOP_GRACE_SECONDS):
        """Stop the server's whole process group: SIGTERM, then SIGKILL after `grace`.

        Signals the group even when the server itself is gone: after a crash its workers
        (e.g. vLLM's EngineCore) can outlive it and keep holding GPU memory.
        """
        if self.proc is None:
            return
        pgid = self.proc.pid  # start_new_session: the server leads its own group
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.5)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                self.proc.poll()  # reap the server so it does not linger as a zombie
                if not _group_alive(pgid):
                    return
                time.sleep(0.05)


def watch(engine, interval, probe_timeout, max_failures):
    """Return the reason the engine stopped serving; runs until then."""
    failures = 0
    while True:
        # Sleep in short steps so a dead process is noticed within a second.
        for _ in range(max(1, int(interval))):
            code = engine.exit_code()
            if code is not None:
                return f"engine exited with code {code}"
            time.sleep(1)
        if engine.healthy(probe_timeout):
            failures = 0
            continue
        failures += 1
        log(f"health probe failed ({failures}/{max_failures})")
        if failures >= max_failures:
            return f"engine failed {failures} consecutive health probes"


def _watchdog(client, engine, interval, probe_timeout, max_failures):
    reason = watch(engine, interval, probe_timeout, max_failures)
    if engine.stopping.is_set():
        return
    log(f"{reason}; recycling the execution environment")
    client.report_error("exit", "Extension.EngineUnhealthy", reason)
    engine.stop()
    os._exit(1)


def _env_int(env, name, default):
    value = int(env.get(name, "") or default)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def run(engine_name, extension_name, env=os.environ):
    if engine_name not in ENGINES:
        log(f"unknown engine {engine_name!r}; expected one of {sorted(ENGINES)}")
        return 2
    build_command, prefix, default_probe_timeout = ENGINES[engine_name]

    client = ExtensionsClient(env["AWS_LAMBDA_RUNTIME_API"], extension_name)
    client.register()
    log(f"registered as {extension_name}")

    engine = None
    if env.get("ENGINE_EXTENSION", "on").lower() in ("off", "false", "0", "disabled"):
        log("ENGINE_EXTENSION is off; not starting an engine")
    else:
        port = env.get(f"{prefix}_SERVER_PORT", "8000")
        try:
            # Inside the try: a bad setting (unbalanced quote in *_SERVER_ARGS, a non-integer
            # timeout) is reported as an init error naming it, not as a bare crash.
            probe_timeout = _env_int(env, "ENGINE_HEALTH_TIMEOUT", default_probe_timeout)
            interval = _env_int(env, "ENGINE_HEALTH_INTERVAL", 10)
            max_failures = _env_int(env, "ENGINE_HEALTH_FAILURES", 3)
            startup_timeout = _env_int(env, f"{prefix}_SERVER_TIMEOUT", 600)
            engine = Engine(build_command(env), f"http://127.0.0.1:{port}/health")
            engine.start()
            engine.wait_ready(startup_timeout, probe_timeout)
        except Exception as e:
            log(f"init failed: {e}")
            client.report_error("init", "Extension.EngineInitFailed", str(e))
            if engine is not None:
                engine.stop()
            return 1
        log("engine is healthy")
        threading.Thread(
            target=_watchdog,
            args=(client, engine, interval, probe_timeout, max_failures),
            daemon=True,
        ).start()

    try:
        while True:
            event = client.next_event()
            if event.get("eventType") == "SHUTDOWN":
                log(f"shutdown: {event.get('shutdownReason', 'unknown')}")
                return 0
    finally:
        # Also on an Extensions API error: the engine runs in its own session and would
        # otherwise be orphaned, still holding GPU memory.
        if engine is not None:
            engine.stopping.set()
            engine.stop()


def main(argv):
    if len(argv) != 3:
        log("usage: engine_extension.py ENGINE EXTENSION_NAME")
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
