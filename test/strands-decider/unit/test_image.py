"""The strands-decider image: the locked environment, the kernels and the SageMaker launcher.

Runs inside the image on CPU. The expected versions come from the image config (environment) and from
docker/strands-decider/pyproject.toml, the image's dependency pins.
"""

import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO = Path(__file__).resolve().parents[3]
PYPROJECT = tomllib.loads((REPO / "docker/strands-decider/pyproject.toml").read_text())
PINS = {
    req.name: spec.version
    for req in map(Requirement, PYPROJECT["project"]["dependencies"])
    for spec in req.specifier
    if spec.operator == "=="
}
ENTRYPOINT = "/usr/local/bin/sagemaker_entrypoint.sh"


def test_python_version():
    expected = os.environ["EXPECTED_PYTHON_VERSION"]
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == expected


def test_strands_decider_is_the_configured_version():
    # A build from a tagged commit reports its release (0.3.0); one between tags, a local version
    # (0.3.1.dev2+g1234abc), whose public part the config holds.
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
    assert os.environ["DECIDER_MAX_CONCURRENCY"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "DECIDER_CHECKPOINT" not in os.environ  # found under /opt/ml/model at start
    assert "DECIDER_DEVICE_MAP" not in os.environ


def test_telemetry_reports_this_framework():
    script = Path("/usr/local/bin/bash_telemetry.sh").read_text()
    assert '--framework "strands_decider"' in script
    assert os.environ["DLC_CONTAINER_TYPE"] == "inference"
    assert f'--framework-version "{os.environ["EXPECTED_FRAMEWORK_VERSION"]}"' in script


def test_oss_attribution_lists_every_python_package_and_tqdm_source():
    licenses = Path("/root/PYTHON_PACKAGES_LICENSES").read_text()
    for package in ("torch", "causal_conv1d", "strands-decider"):
        assert f"Package Name: {package}\n" in licenses
    # tqdm is MPL-2.0 AND MIT: its source URL must be listed for upload.
    sources = Path("/root/THIRD_PARTY_SOURCE_CODE_URLS").read_text().split("\n")
    version = importlib.metadata.version("tqdm")
    assert (
        f"tqdm {version} https://github.com/tqdm/tqdm/archive/refs/tags/v{version}.tar.gz"
        in sources
    )


def test_entrypoint_is_executable():
    assert os.access(ENTRYPOINT, os.X_OK)


def test_serve_has_the_options_the_entrypoint_uses():
    out = subprocess.run(
        ["strands-decider", "serve", "--help"], capture_output=True, text=True, check=True
    ).stdout
    for option in (
        "--host",
        "--port",
        "--device",
        "--max-batch",
        "--no-warmup",
        "--device-map",
        "--max-concurrency",
    ):
        # Whole options: --device must not pass because of --device-map.
        assert re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", out), option


def _entrypoint(tmp_path, *command, **env):
    """Run the entrypoint with a stand-in `strands-decider` that prints its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "strands-decider"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')  # one argument per line
    fake.chmod(0o755)
    model_dir = tmp_path / "model"
    run_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SM_MODEL_DIR": str(model_dir),
        **env,
    }
    return subprocess.run([ENTRYPOINT, *command], capture_output=True, text=True, env=run_env)


def _checkpoint(path, base_model="org/base-model"):
    path.mkdir(parents=True)
    (path / "strands_decider_config.json").write_text(json.dumps({"base_model": str(base_model)}))
    return path


def _entrypoint_command(tmp_path, *command, **env):
    """The `strands-decider` arguments the entrypoint runs, one per item."""
    result = _entrypoint(tmp_path, *(command or ("serve",)), **env)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_entrypoint_serves_the_sagemaker_contract_by_default(tmp_path):
    _checkpoint(tmp_path / "model")
    args = _entrypoint_command(tmp_path)
    assert args[:2] == ["serve", str(tmp_path / "model")]
    assert args[args.index("--port") + 1] == "8080"
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--device") + 1] == "cuda"
    assert "--no-warmup" not in args and "--device-map" not in args
    assert args[args.index("--max-concurrency") + 1] == "1"


def test_entrypoint_finds_the_checkpoint_next_to_its_base_model(tmp_path):
    # The layout of a model tarball: the checkpoint folder and base/, the base model it reads.
    base = tmp_path / "model" / "base"
    checkpoint = _checkpoint(tmp_path / "model" / "strands-decider-2B-qwen3.5-v1-2610", base)
    base.mkdir()
    (base / "config.json").write_text("{}")
    assert _entrypoint_command(tmp_path)[1] == str(checkpoint)


def test_entrypoint_refuses_a_checkpoint_that_does_not_name_its_base_folder(tmp_path):
    # A published checkpoint names its base model's Hub id, which can't load offline.
    _checkpoint(tmp_path / "model" / "strands-decider-2B-qwen3.5-v1-2610", "Qwen/Qwen3.5-2B-Base")
    (tmp_path / "model" / "base").mkdir()
    result = _entrypoint(tmp_path, "serve")
    assert result.returncode != 0
    assert "base_model" in result.stderr and str(tmp_path / "model" / "base") in result.stderr


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
        DECIDER_MAX_CONCURRENCY="4",
        DECIDER_DEVICE_MAP="auto",
        DECIDER_ARGS='--max-batch 8 --model-name "fraud triage" --tag *',
    )
    assert args[1] == "/models/mine"
    assert "--no-warmup" in args
    assert args[args.index("--max-concurrency") + 1] == "4"
    assert args[args.index("--device-map") + 1] == "auto"
    assert args[args.index("--max-batch") + 1] == "8"
    assert args[args.index("--model-name") + 1] == "fraud triage"  # quotes group, and aren't kept
    assert args[args.index("--tag") + 1] == "*"  # not expanded to file names


def test_entrypoint_runs_without_a_cap_when_max_concurrency_is_0(tmp_path):
    args = _entrypoint_command(
        tmp_path, DECIDER_CHECKPOINT="/models/mine", DECIDER_MAX_CONCURRENCY="0"
    )
    assert "--max-concurrency" not in args


def test_entrypoint_forwards_options_after_serve(tmp_path):
    args = _entrypoint_command(
        tmp_path, "serve", "--device-map", "auto", DECIDER_CHECKPOINT="/models/mine"
    )
    assert args[-2:] == ["--device-map", "auto"]


def test_entrypoint_takes_leading_options_as_serve_options(tmp_path):
    args = _entrypoint_command(tmp_path, "--max-batch", "8", DECIDER_CHECKPOINT="/models/mine")
    assert args[:2] == ["serve", "/models/mine"]
    assert args[-2:] == ["--max-batch", "8"]


def test_entrypoint_runs_any_other_command(tmp_path):
    result = _entrypoint(tmp_path, "echo", "hello")
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
