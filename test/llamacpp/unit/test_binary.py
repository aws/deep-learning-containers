"""Verify llama-server binary and shared libraries are installed correctly."""

import os
import subprocess

import pytest


class TestLlamaServerBinary:
    """Verify llama-server binary exists and is functional."""

    LLAMA_SERVER = "/usr/local/bin/llama-server"

    def test_binary_exists(self):
        assert os.path.isfile(self.LLAMA_SERVER), f"{self.LLAMA_SERVER} not found"

    def test_binary_is_executable(self):
        assert os.access(self.LLAMA_SERVER, os.X_OK), f"{self.LLAMA_SERVER} is not executable"

    def test_help_returns_zero(self):
        result = subprocess.run(
            [self.LLAMA_SERVER, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"--help failed: {result.stderr}"
        assert "usage" in result.stdout.lower() or "llama" in result.stdout.lower(), (
            f"Unexpected --help output: {result.stdout[:200]}"
        )


class TestSharedLibraries:
    """Verify llama.cpp shared libraries are present and discoverable."""

    @pytest.mark.parametrize(
        "pattern",
        ["libggml*.so", "libllama*.so"],
    )
    def test_shared_lib_exists(self, pattern):
        import glob

        matches = glob.glob(f"/usr/local/lib/{pattern}")
        assert matches, f"No files matching {pattern} found in /usr/local/lib/"

    def test_ldd_resolves_libs(self):
        result = subprocess.run(
            ["ldd", "/usr/local/bin/llama-server"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"ldd failed: {result.stderr}"
        output = result.stdout
        assert "libllama.so" in output, "libllama.so not resolved by ldd"
        assert "libggml.so" in output, "libggml.so not resolved by ldd"
        # libcuda.so.1 is expected to be "not found" without GPU driver
        for lib in ["libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"]:
            assert f"{lib} => not found" not in output, f"{lib} not resolved"
