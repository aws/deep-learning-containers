"""Verify environment variables, binaries, and the NCCL/EFA setup."""

import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "name, value",
    [
        ("DLC_CONTAINER_TYPE", "training"),
        ("MILES_MOONCAKE_AVAILABLE", "1"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("SGLANG_RUST_BUILD_MODE", "never"),
        ("VIRTUAL_ENV", "/opt/venv"),
    ],
)
def test_env_var(name, value):
    assert os.environ.get(name) == value


@pytest.mark.parametrize(
    "directory",
    ["/opt/venv/bin", "/opt/amazon/efa/bin", "/opt/amazon/openmpi/bin", "/usr/local/cuda/bin"],
)
def test_path_contains(directory):
    assert directory in os.environ["PATH"].split(":"), f"{directory} not in PATH"


@pytest.mark.parametrize(
    "binary",
    [
        "all_reduce_perf",
        "ffmpeg",
        "fi_info",
        "mooncake_master",
        "mpirun",
        "nvcc",
        "sgl-model-gateway",
    ],
)
def test_binary_on_path(binary):
    assert shutil.which(binary) is not None, f"{binary} not found on PATH"


# Diagnostic and utility tools the upstream Miles image ships.
@pytest.mark.parametrize(
    "tool",
    [
        "curl",
        "dig",
        "ethtool",
        "git",
        "jq",
        "patch",
        "patchelf",
        "perl",
        "rsync",
        "tmux",
        "unzip",
        "wget",
    ],
)
def test_tool_on_path(tool):
    assert shutil.which(tool) is not None, f"{tool} not found on PATH"


# Miles' Kubernetes launcher and workbench drive runs with these clients.
@pytest.mark.parametrize(
    "command", [["kubectl", "version", "--client"], ["helm", "version", "--short"]]
)
def test_kubernetes_client_runs(command):
    subprocess.run(command, check=True, capture_output=True)


def test_ffmpeg_runs():
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)


def test_ofi_nccl_plugin_installed():
    assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net-ofi.so")
