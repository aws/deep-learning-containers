"""Verify framework versions match the build's source manifest."""

import os
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
