"""Verify the image entrypoint launches commands and sets the NCCL interface."""

import os
import subprocess

ENTRYPOINT = "/usr/local/bin/miles-entrypoint"


def run_entrypoint(*command, env=None):
    return subprocess.run(
        [ENTRYPOINT, *command], capture_output=True, text=True, env=env, check=False
    )


def last_line(text):
    return text.strip().splitlines()[-1]


def test_entrypoint_runs_command():
    # CI runs unit tests without a GPU, which covers hosts with no NVIDIA driver.
    result = run_entrypoint("python", "-c", "print('ok')")
    assert result.returncode == 0, result.stderr
    assert last_line(result.stdout) == "ok"


def test_entrypoint_defaults_nccl_socket_ifname():
    env = {key: value for key, value in os.environ.items() if key != "NCCL_SOCKET_IFNAME"}
    result = run_entrypoint("bash", "-c", 'echo "${NCCL_SOCKET_IFNAME}"', env=env)
    assert last_line(result.stdout) == "^docker0,lo"


def test_entrypoint_keeps_explicit_nccl_socket_ifname():
    env = {**os.environ, "NCCL_SOCKET_IFNAME": "eth0"}
    result = run_entrypoint("bash", "-c", 'echo "${NCCL_SOCKET_IFNAME}"', env=env)
    assert last_line(result.stdout) == "eth0"
