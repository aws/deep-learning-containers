"""Verify pytorch-image-specific environment — pytorch image."""

import os
import shutil
import subprocess

import pytest


def test_ld_library_path_includes_usr_local_lib():
    """/usr/local/lib must be in LD_LIBRARY_PATH for FFmpeg shared libs."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    assert "/usr/local/lib" in ld


def test_nvidia_driver_capabilities():
    assert os.environ.get("NVIDIA_DRIVER_CAPABILITIES") == "compute,utility,video"


@pytest.mark.parametrize(
    "var", ["TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
)
def test_model_and_jit_caches_under_tmp(var):
    """Model downloads and inductor/triton JIT output must land in /tmp."""
    value = os.environ.get(var, "")
    assert value.startswith("/tmp/"), f"{var}={value!r}"


@pytest.mark.parametrize("tool", ["gcc", "which"])
def test_host_compiler_on_path(tool):
    """Triton and Inductor shell out to a C compiler, so one must be installed."""
    assert shutil.which(tool), f"{tool} not found on PATH"


def test_gcc_compiles_and_links(tmp_path):
    """Presence on PATH is not enough — the toolchain must actually produce a binary."""
    src = tmp_path / "probe.c"
    src.write_text("int main(void) { return 0; }\n")
    out = tmp_path / "probe"
    subprocess.run(["gcc", str(src), "-o", str(out)], check=True, capture_output=True)
    subprocess.run([str(out)], check=True)
