"""Verify only the cu13 nixl variant is installed and its .so links libcudart 13."""

import subprocess
from pathlib import Path

import pytest


def _pip_list():
    return subprocess.run(
        ["pip", "list", "--format=freeze"], capture_output=True, text=True, check=True
    ).stdout.lower()


def test_nixl_cu13_installed():
    assert "nixl-cu13" in _pip_list()


def test_nixl_cu12_not_installed():
    assert "nixl-cu12" not in _pip_list()


def test_nixl_ep_so_links_cudart_13():
    candidates = list(Path("/opt/venv/lib").rglob("nixl_ep_cpp*.so"))
    assert candidates, "nixl_ep_cpp .so not found under /opt/venv/lib"
    ldd = subprocess.run(["ldd", str(candidates[0])], capture_output=True, text=True).stdout
    assert "libcudart.so.13" in ldd
    assert "libcudart.so.12" not in ldd


def test_torch_links_cudart_13():
    import torch

    torch_lib = Path(torch.__file__).parent / "lib"
    torch_so = torch_lib / "libtorch_cuda.so"
    if not torch_so.exists():
        pytest.skip("libtorch_cuda.so not found; layout may have changed")
    ldd = subprocess.run(["ldd", str(torch_so)], capture_output=True, text=True).stdout
    assert "libcudart.so.13" in ldd
    assert "libcudart.so.12" not in ldd
