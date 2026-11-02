"""The strands-decider image: the locked environment, the kernels and the SageMaker launcher.

Runs inside the image on CPU. The expected versions come from the image config (environment) and from
docker/strands-decider/pyproject.toml, the image's dependency pins.
"""

import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = tomllib.loads((REPO / "docker/strands-decider/pyproject.toml").read_text())
PINS = dict(
    re.match(r"([A-Za-z0-9_.-]+)==(\S+)", dep).groups()
    for dep in PYPROJECT["project"]["dependencies"]
    if "==" in dep
)
ENTRYPOINT = "/usr/local/bin/sagemaker_entrypoint.sh"
LAUNCHER = "/usr/local/bin/sagemaker_serve.py"


def test_python_version():
    expected = os.environ["EXPECTED_PYTHON_VERSION"]
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == expected


def test_strands_decider_is_the_configured_version():
    # A build from a commit reports a local version (0.2.1.dev8+g13faf52); the config holds the public part.
    installed = importlib.metadata.version("strands-decider")
    assert installed.split("+")[0] == os.environ["EXPECTED_FRAMEWORK_VERSION"]


@pytest.mark.parametrize("package", sorted(PINS))
def test_pinned_packages_are_installed_at_their_pin(package):
    installed = importlib.metadata.version(package)
    assert installed.split("+")[0] == PINS[package]


def test_torch_is_a_cuda_build():
    import torch

    assert torch.version.cuda is not None


def test_causal_conv1d_is_the_compiled_kernel():
    import causal_conv1d
    import causal_conv1d_cuda  # noqa: F401  the compiled extension, not the PyTorch fallback

    assert causal_conv1d.__version__ == os.environ["EXPECTED_CAUSAL_CONV1D_VERSION"]


def test_flash_linear_attention_is_installed():
    # Importing it needs a GPU (Triton looks for a driver); the GPU tests run its kernels on the Qwen3.5 model.
    assert importlib.metadata.version("flash-linear-attention")


def test_triton_can_compile_its_launcher():
    # Triton builds a small C launcher at run time; without a C compiler the first GPU request fails.
    assert subprocess.run(["gcc", "--version"], capture_output=True).returncode == 0


def test_sagemaker_defaults():
    assert os.environ["DECIDER_CHECKPOINT"] == "/opt/ml/model"
    assert os.environ["DECIDER_DEVICE"] == "cuda"
    assert os.environ["DECIDER_WARMUP"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "DECIDER_DEVICE_MAP" not in os.environ


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT, os.X_OK)


def test_launcher_serves_the_upstream_cli():
    # The launcher wraps `strands-decider serve`; every upstream option still reaches it.
    out = subprocess.run(
        [sys.executable, LAUNCHER, "serve", "--help"], capture_output=True, text=True, check=True
    ).stdout
    for option in ("--host", "--port", "--device", "--max-batch", "--strict-window"):
        assert option in out
