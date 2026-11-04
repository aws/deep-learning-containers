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
    assert os.environ["DECIDER_DEVICE"] == "cuda"
    assert os.environ["DECIDER_WARMUP"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "DECIDER_CHECKPOINT" not in os.environ  # found under /opt/ml/model at start
    assert "DECIDER_DEVICE_MAP" not in os.environ


def test_telemetry_reports_this_framework():
    script = Path("/usr/local/bin/bash_telemetry.sh").read_text()
    assert '--framework "strands_decider"' in script
    assert f'--framework-version "{os.environ["EXPECTED_FRAMEWORK_VERSION"]}"' in script


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT, os.X_OK)


def test_serve_has_the_options_the_entrypoint_uses():
    out = subprocess.run(
        ["strands-decider", "serve", "--help"], capture_output=True, text=True, check=True
    ).stdout
    for option in ("--host", "--port", "--device", "--max-batch", "--no-warmup", "--device-map"):
        assert option in out


def _entrypoint(tmp_path, *command, **env):
    """Run the entrypoint with a stand-in `strands-decider` that prints its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "strands-decider"
    fake.write_text('#!/bin/bash\necho "$@"\n')
    fake.chmod(0o755)
    model_dir = tmp_path / "model"
    run_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SM_MODEL_DIR": str(model_dir),
        **env,
    }
    return subprocess.run([ENTRYPOINT, *command], capture_output=True, text=True, env=run_env)


def _checkpoint(path):
    path.mkdir(parents=True)
    (path / "strands_decider_config.json").write_text("{}")
    return path


def _entrypoint_command(tmp_path, **env):
    """The `strands-decider` command line the entrypoint runs for `serve`."""
    result = _entrypoint(tmp_path, "serve", **env)
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_entrypoint_serves_the_sagemaker_contract_by_default(tmp_path):
    _checkpoint(tmp_path / "model")
    args = _entrypoint_command(tmp_path)
    assert args[:2] == ["serve", str(tmp_path / "model")]
    assert args[args.index("--port") + 1] == "8080"
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--device") + 1] == "cuda"
    assert "--no-warmup" not in args and "--device-map" not in args


def test_entrypoint_finds_the_checkpoint_next_to_its_base_model(tmp_path):
    # The layout of a model tarball: the checkpoint folder and base/, the base model it reads.
    checkpoint = _checkpoint(tmp_path / "model" / "strands-decider-2B-qwen3.5-v1-2610")
    (tmp_path / "model" / "base").mkdir()
    (tmp_path / "model" / "base" / "config.json").write_text("{}")
    assert _entrypoint_command(tmp_path)[1] == str(checkpoint)


def test_entrypoint_refuses_an_unclear_model_folder(tmp_path):
    _checkpoint(tmp_path / "model" / "a")
    _checkpoint(tmp_path / "model" / "b")
    result = _entrypoint(tmp_path, "serve")
    assert result.returncode != 0
    assert "DECIDER_CHECKPOINT" in result.stderr


def test_entrypoint_maps_the_environment_to_serve_options(tmp_path):
    args = _entrypoint_command(
        tmp_path,
        DECIDER_CHECKPOINT="/models/mine",
        DECIDER_WARMUP="0",
        DECIDER_DEVICE_MAP="auto",
        DECIDER_ARGS="--max-batch 8 --model-name *",
    )
    assert args[1] == "/models/mine"
    assert "--no-warmup" in args
    assert args[args.index("--device-map") + 1] == "auto"
    assert args[args.index("--max-batch") + 1] == "8"
    assert args[args.index("--model-name") + 1] == "*"  # not expanded to file names


def test_entrypoint_runs_any_other_command(tmp_path):
    result = _entrypoint(tmp_path, "echo", "hello")
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
