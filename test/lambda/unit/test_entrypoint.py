"""Entrypoint dispatch smoke tests.

The entrypoints pick between the Runtime Interface Emulator and a direct exec on
AWS_LAMBDA_RUNTIME_API. The exec branch runs against the shipped script; the RIE
branch runs a copy with the RIE path swapped for a recording stub, so these tests
stay CPU-only and never start a real server.
"""

import os
import stat
import subprocess

import pytest

ENTRYPOINTS = ["/lambda_entrypoint.sh", "/sglang_entrypoint.sh", "/vllm_entrypoint.sh"]
RIE_PATH = "/usr/local/bin/aws-lambda-rie"


@pytest.fixture(scope="module")
def entrypoint():
    script = next((s for s in ENTRYPOINTS if os.path.isfile(s)), None)
    if script is None:
        pytest.fail(f"no entrypoint found (looked for {ENTRYPOINTS})")
    return script


def _stubbed(entrypoint, tmp_path):
    record = tmp_path / "argv.txt"
    stub = tmp_path / "rie-stub.sh"
    stub.write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a" >> {record}; done\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    src = open(entrypoint).read()
    assert RIE_PATH in src, f"{entrypoint} no longer references {RIE_PATH}"
    copy = tmp_path / "entrypoint.sh"
    copy.write_text(src.replace(RIE_PATH, str(stub)))
    copy.chmod(copy.stat().st_mode | stat.S_IEXEC)
    return copy, record


def _env_without_runtime_api():
    return {k: v for k, v in os.environ.items() if k != "AWS_LAMBDA_RUNTIME_API"}


def test_entrypoint_executable(entrypoint):
    assert os.access(entrypoint, os.X_OK), f"{entrypoint} not executable"


def test_rie_binary_referenced_by_entrypoint_exists(entrypoint):
    assert RIE_PATH in open(entrypoint).read()
    assert os.access(RIE_PATH, os.X_OK), f"{RIE_PATH} missing or not executable"


def test_execs_command_when_runtime_api_set(entrypoint):
    env = {**os.environ, "AWS_LAMBDA_RUNTIME_API": "127.0.0.1:9001"}
    out = subprocess.run(
        [entrypoint, "/bin/echo", "smoke"], env=env, capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "smoke"


def test_replaces_shell_so_the_ric_owns_pid_1(entrypoint):
    env = {**os.environ, "AWS_LAMBDA_RUNTIME_API": "127.0.0.1:9001"}
    proc = subprocess.Popen(
        [entrypoint, "sh", "-c", "echo $$"], env=env, stdout=subprocess.PIPE, text=True
    )
    out, _ = proc.communicate(timeout=30)
    assert int(out.strip()) == proc.pid, "entrypoint forked instead of exec'ing the command"


def test_uses_rie_when_runtime_api_unset(entrypoint, tmp_path):
    copy, record = _stubbed(entrypoint, tmp_path)
    out = subprocess.run(
        [str(copy), "python", "-m", "awslambdaric", "handler.handler"],
        env=_env_without_runtime_api(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert record.read_text().split() == ["python", "-m", "awslambdaric", "handler.handler"]


def test_rie_branch_preserves_argument_boundaries(entrypoint, tmp_path):
    copy, record = _stubbed(entrypoint, tmp_path)
    subprocess.run(
        [str(copy), "one two three"], env=_env_without_runtime_api(), check=True, timeout=30
    )
    assert record.read_text().splitlines() == ["one two three"]
