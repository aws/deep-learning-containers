"""
Sanity tests for llama.cpp DLC images.

Run inside the container with GPU:
    docker run --rm --gpus all --entrypoint python3 <image> /tests/test_sanity_llamacpp.py

Or with pytest:
    docker run --rm --entrypoint pytest <image> /tests/test_sanity_llamacpp.py -v

Test categories:
    1. TestBinaryHealth       - llama-server binary exists, is executable, --help works
    2. TestEntrypointContract - entrypoint exists, is executable, references llama-server
"""

import os
import subprocess
import unittest


class TestBinaryHealth(unittest.TestCase):
    """Category 1: Verify llama-server binary is installed and functional."""

    LLAMA_SERVER = "/usr/local/bin/llama-server"

    def test_binary_exists(self):
        """llama-server must exist at the expected path."""
        self.assertTrue(os.path.isfile(self.LLAMA_SERVER), f"{self.LLAMA_SERVER} not found")

    def test_binary_is_executable(self):
        """llama-server must be executable."""
        self.assertTrue(
            os.access(self.LLAMA_SERVER, os.X_OK), f"{self.LLAMA_SERVER} is not executable"
        )

    def test_help_works(self):
        """llama-server --help must return 0 and produce recognizable output."""
        result = subprocess.run(
            [self.LLAMA_SERVER, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, f"--help failed: {result.stderr}")
        combined = (result.stdout + result.stderr).lower()
        self.assertTrue(
            "usage" in combined or "llama" in combined,
            f"Unexpected --help output: {result.stdout[:200]}",
        )


class TestEntrypointContract(unittest.TestCase):
    """Category 2: Verify entrypoint structural correctness."""

    ENTRYPOINT_CANDIDATES = [
        "/usr/local/bin/sagemaker_entrypoint.sh",
        "/usr/local/bin/dockerd_entrypoint.sh",
        "/usr/bin/serve",
    ]

    def _find_entrypoint(self):
        for path in self.ENTRYPOINT_CANDIDATES:
            if os.path.isfile(path):
                return path
        return None

    def test_entrypoint_exists(self):
        """At least one entrypoint must be present."""
        ep = self._find_entrypoint()
        self.assertIsNotNone(ep, f"No entrypoint found at any of {self.ENTRYPOINT_CANDIDATES}")

    def test_entrypoint_is_executable(self):
        """Entrypoint must be executable."""
        ep = self._find_entrypoint()
        if not ep:
            self.skipTest("No entrypoint found")
        self.assertTrue(os.access(ep, os.X_OK), f"{ep} is not executable")

    def test_entrypoint_references_llama_server(self):
        """Entrypoint must reference llama-server."""
        ep = self._find_entrypoint()
        if not ep:
            self.skipTest("No entrypoint found")
        with open(ep) as f:
            content = f.read()
        self.assertIn(
            "llama-server",
            content,
            "Entrypoint does not reference llama-server",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
