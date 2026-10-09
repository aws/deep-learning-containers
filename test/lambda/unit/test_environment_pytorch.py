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


@pytest.mark.parametrize("tool", ["gcc", "g++", "which"])
def test_host_compiler_on_path(tool):
    """Triton resolves gcc via shutil.which; Inductor's cpp_wrapper resolves g++."""
    assert shutil.which(tool), f"{tool} not found on PATH"


C_PROBE = "int main(void) { return 0; }\n"
CPP_PROBE = "#include <vector>\nint main() { return std::vector<int>{}.size(); }\n"


@pytest.mark.parametrize(
    ("compiler", "ext", "source"),
    [("gcc", "c", C_PROBE), ("g++", "cpp", CPP_PROBE)],
)
def test_toolchain_compiles_and_links(tmp_path, compiler, ext, source):
    """Presence on PATH is not enough — the toolchain must actually produce a binary."""
    src = tmp_path / f"probe.{ext}"
    src.write_text(source)
    out = tmp_path / f"probe_{ext}"
    subprocess.run([compiler, str(src), "-o", str(out)], check=True, capture_output=True)
    subprocess.run([str(out)], check=True)
