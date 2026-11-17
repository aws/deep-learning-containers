"""Verify framework versions match the build's source manifest."""

import os
import subprocess
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from packaging.version import Version

SOURCE_MANIFEST = Path("/opt/aws/dlc/SOURCE_MANIFEST")


@pytest.fixture(scope="module")
def manifest():
    values = {}
    for line in SOURCE_MANIFEST.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key and value:
            values[key] = value
    return values


def test_miles_version(manifest):
    assert version("miles") == manifest["framework_version"]


def test_sglang_version(manifest):
    assert version("sglang") == manifest["sglang_version"]


def test_torch_version(manifest):
    assert Version(torch.__version__).base_version == manifest["torch_version"]


def test_torch_cuda_version(manifest):
    assert torch.version.cuda == ".".join(manifest["cuda_version"].split(".")[:2])


def test_sglang_kernel_is_cuda_build(manifest):
    cuda_tag = "cu" + "".join(manifest["cuda_version"].split(".")[:2])
    assert Version(version("sglang-kernel")).local == cuda_tag


def test_supported_cuda_arches_match_build(manifest):
    expected = manifest["torch_cuda_arch_list"].replace(";", ",")
    assert os.environ["MILES_SUPPORTED_CUDA_ARCHES"] == expected


def git(checkout, *args):
    return subprocess.run(
        ["git", "-C", checkout, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


# Miles reports `git diff` of editable checkouts in its environment report, so
# each shipped checkout must sit at its pinned commit with nothing changed.
@pytest.mark.parametrize(
    "checkout, key",
    [
        ("/opt/miles", "miles"),
        ("/opt/sglang", "sglang"),
        ("/opt/Megatron-LM", "megatron_core"),
        ("/opt/torchtitan", "torchtitan"),
    ],
)
def test_source_checkout_is_clean_at_pinned_commit(manifest, checkout, key):
    assert git(checkout, "rev-parse", "HEAD") == manifest[key]
    assert git(checkout, "diff", "--stat", "HEAD") == ""
    assert git(checkout, "ls-files", "--others", "--exclude-standard") == ""
