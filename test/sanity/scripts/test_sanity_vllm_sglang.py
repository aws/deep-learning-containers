"""
GPU-free sanity tests for vLLM and SGLang DLC images.

Run inside the container:
    docker run --rm --entrypoint python3 <image> /tests/test_sanity_vllm_sglang.py

Or with pytest:
    docker run --rm --entrypoint pytest <image> /tests/test_sanity_vllm_sglang.py -v

Optionally pass IMAGE_TAG env var for version consistency checks:
    -e IMAGE_TAG="vllm:0.17-gpu-py312-cu129-ubuntu22.04-sagemaker-v1"

Test categories:
    1. TestCudaJitDependencies  - CUDA binaries required by JIT libraries (cuobjdump, nvcc, ptxas, etc.)
    2. TestEntrypointArgHandling - SageMaker entrypoint env var → CLI arg translation
                                  (boolean flags, model auto-detection, HF_MODEL_ID fallback)
    3. TestPackageVersionConsistency - Image tag vs installed versions, torch↔CUDA agreement
    4. TestEntrypointContract - Entrypoint exists, is executable, invokes correct server

Framework-aware: tests auto-detect whether the image is vLLM or SGLang based on the
entrypoint content (SM_VLLM_ vs SM_SGLANG_ prefix). Tests that don't apply to a given
image flavor (e.g. SageMaker entrypoint tests on an EC2 image) are skipped automatically.
"""

import json
import os
import re
import subprocess
import sys
import unittest


class TestCudaJitDependencies(unittest.TestCase):
    """Category 1: Verify CUDA binaries required by JIT-compiling libraries exist."""

    # Map of binary -> list of libraries that need it
    REQUIRED_CUDA_BINARIES = {
        "nvcc": ["deep_gemm JIT", "flashinfer JIT"],
        "ptxas": ["deep_gemm JIT", "vllm custom ops", "torch inductor"],
        "cuobjdump": ["deep_gemm JIT (SASS extraction for shared memory analysis)"],
        "fatbinary": ["nvcc toolchain"],
        "nvlink": ["nvcc toolchain"],
    }

    def _find_cuda_home(self):
        """Find CUDA home the same way deep_gemm does."""
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            return cuda_home
        # Check common locations
        for path in ["/usr/local/cuda", "/usr/local/cuda-12"]:
            if os.path.isdir(path):
                return path
        self.fail("Cannot find CUDA_HOME")

    def test_cuda_binaries_present(self):
        """All CUDA binaries needed by JIT libraries must exist in the image."""
        cuda_home = self._find_cuda_home()
        bin_dir = os.path.join(cuda_home, "bin")
        missing = {}
        for binary, consumers in self.REQUIRED_CUDA_BINARIES.items():
            if not os.path.isfile(os.path.join(bin_dir, binary)):
                missing[binary] = consumers
        if missing:
            lines = [f"Missing CUDA binaries in {bin_dir}:"]
            for binary, consumers in missing.items():
                lines.append(f"  - {binary} (needed by: {', '.join(consumers)})")
            self.fail("\n".join(lines))

    def test_nvcc_runs(self):
        """nvcc must be executable and return a version string."""
        cuda_home = self._find_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, f"nvcc failed: {result.stderr}")
        self.assertIn("release", result.stdout.lower())

    def test_deep_gemm_imports(self):
        """deep_gemm must import without error."""
        try:
            # Upstream now vendors DeepGEMM under vllm.third_party in the vLLM wheel
            # (see cmake/external_projects/deepgemm.cmake). Fall back to top-level
            # import for older builds that installed it as a separate wheel.
            try:
                from vllm.third_party import deep_gemm  # noqa: F811
            except ImportError:
                import deep_gemm  # noqa: F811
        except ImportError as e:
            if "libcuda" in str(e):
                self.skipTest("libcuda not available (no GPU in test environment)")
            raise
        self.assertTrue(hasattr(deep_gemm, "__version__"))

    def test_flashinfer_imports(self):
        """flashinfer must import without error."""
        import flashinfer  # noqa: F811

        self.assertTrue(hasattr(flashinfer, "__version__"))

    def test_triton_imports(self):
        """triton must import without error."""
        import triton  # noqa: F811

        self.assertTrue(hasattr(triton, "__version__"))


class TestEntrypointArgHandling(unittest.TestCase):
    """Category 2: Verify sagemaker_entrypoint.sh handles env vars correctly."""

    SAGEMAKER_ENTRYPOINT = None  # auto-detected in setUp

    ENTRYPOINT_CANDIDATES = [
        "/usr/local/bin/sagemaker_entrypoint.sh",
        "/usr/bin/serve",
    ]

    def setUp(self):
        # Find the sagemaker entrypoint
        for path in self.ENTRYPOINT_CANDIDATES:
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                if "SM_VLLM_" in content or "SM_SGLANG_" in content:
                    self.SAGEMAKER_ENTRYPOINT = path
                    break
        if not self.SAGEMAKER_ENTRYPOINT:
            self.skipTest("No SageMaker entrypoint found")
        with open(self.SAGEMAKER_ENTRYPOINT) as f:
            content = f.read()
        # Detect framework from entrypoint content
        if "SM_SGLANG_" in content:
            self.prefix = "SM_SGLANG_"
            self.model_key = "SM_SGLANG_MODEL_PATH"
            self.model_flag = "--model-path"
        else:
            self.prefix = "SM_VLLM_"
            self.model_key = "SM_VLLM_MODEL"
            self.model_flag = "--model"

    def _get_args(self, env_vars, mount_model_dir=False):
        """Run entrypoint in dry-run mode and capture the generated args."""
        with open(self.SAGEMAKER_ENTRYPOINT) as f:
            script = f.read()
        script = re.sub(
            r"^exec\s+(standard-supervisor\s+)?python3\s+.*$",
            'echo "__ARGS__${ARGS[@]}__END__"',
            script,
            flags=re.MULTILINE,
        )
        # Also suppress start_cuda_compat.sh if present
        script = script.replace("bash /usr/local/bin/start_cuda_compat.sh", "true")

        env = {k: v for k, v in os.environ.items()}
        # Clear any existing SM_VLLM_ / SM_SGLANG_ vars
        for k in list(env.keys()):
            if k.startswith("SM_VLLM_") or k.startswith("SM_SGLANG_"):
                del env[k]
        env.update(env_vars)

        if mount_model_dir:
            os.makedirs("/tmp/fake_model", exist_ok=True)
            with open("/tmp/fake_model/config.json", "w") as f:
                f.write("{}")
            script = script.replace("/opt/ml/model", "/tmp/fake_model")

        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        stdout = result.stdout + result.stderr
        match = re.search(r"__ARGS__(.+?)__END__", stdout)
        if not match:
            self.fail(f"Could not capture args. stdout={result.stdout} stderr={result.stderr}")
        return match.group(1).split()

    def _model_env(self, val="x"):
        """Return env dict with model set to avoid unrelated warnings."""
        return {self.model_key: val}

    def test_string_value(self):
        """Model path env var -> --model/--model-path <value>"""
        args = self._get_args({self.model_key: "/some/model/path"})
        self.assertIn(self.model_flag, args)
        idx = args.index(self.model_flag)
        self.assertEqual(args[idx + 1], "/some/model/path")

    def test_numeric_value(self):
        """Numeric env var -> --flag <number>"""
        env = self._model_env()
        env[f"{self.prefix}MAX_MODEL_LEN"] = "4096"
        args = self._get_args(env)
        self.assertIn("--max-model-len", args)
        idx = args.index("--max-model-len")
        self.assertEqual(args[idx + 1], "4096")

    def test_boolean_true_flag(self):
        """SOME_FLAG=true -> --some-flag (no 'true' value appended)"""
        env = self._model_env()
        env[f"{self.prefix}DISABLE_LOG_STATS"] = "true"
        args = self._get_args(env)
        self.assertIn("--disable-log-stats", args)
        idx = args.index("--disable-log-stats")
        if idx + 1 < len(args):
            self.assertNotEqual(
                args[idx + 1], "true", "Boolean flag should not have 'true' as value"
            )

    def test_boolean_True_flag(self):
        """SOME_FLAG=True -> --some-flag (case insensitive)"""
        env = self._model_env()
        env[f"{self.prefix}DISABLE_LOG_STATS"] = "True"
        args = self._get_args(env)
        self.assertIn("--disable-log-stats", args)
        idx = args.index("--disable-log-stats")
        if idx + 1 < len(args):
            self.assertNotEqual(args[idx + 1], "True")

    def test_boolean_false_omitted(self):
        """SOME_FLAG=false -> flag should be omitted entirely"""
        env = self._model_env()
        env[f"{self.prefix}ENABLE_CHUNKED_PREFILL"] = "false"
        args = self._get_args(env)
        self.assertNotIn("--enable-chunked-prefill", args)

    def test_default_port(self):
        """--port 8080 should always be present."""
        args = self._get_args(self._model_env())
        self.assertIn("--port", args)
        idx = args.index("--port")
        self.assertEqual(args[idx + 1], "8080")

    def test_model_autodetect(self):
        """When model env var is unset but /opt/ml/model exists, auto-detect it."""
        if self.prefix == "SM_SGLANG_":
            # SGLang already defaults --model-path to /opt/ml/model
            args = self._get_args({}, mount_model_dir=True)
            self.assertIn("--model-path", args)
        else:
            args = self._get_args({}, mount_model_dir=True)
            self.assertIn("--model", args)

    def test_hf_model_id_fallback(self):
        """When model env var unset and no /opt/ml/model, fall back to HF_MODEL_ID."""
        if self.prefix == "SM_SGLANG_":
            self.skipTest("SGLang defaults to /opt/ml/model, no HF_MODEL_ID fallback needed")
        args = self._get_args({"HF_MODEL_ID": "meta-llama/Llama-3-8B"})
        self.assertIn("--model", args)
        idx = args.index("--model")
        self.assertEqual(args[idx + 1], "meta-llama/Llama-3-8B")


class TestPackageVersionConsistency(unittest.TestCase):
    """Category 3: Verify package versions match image tag and are internally consistent."""

    def _parse_image_tag(self):
        """Parse IMAGE_TAG env var like 'vllm:0.17-gpu-py312-cu129-ubuntu22.04-sagemaker-v1'."""
        tag = os.environ.get("IMAGE_TAG", "")
        if not tag:
            self.skipTest("IMAGE_TAG not set, skipping tag-based checks")
        return tag

    def test_vllm_version_matches_tag(self):
        """vllm.__version__ major.minor should match the image tag."""
        tag = self._parse_image_tag()
        match = re.search(r"vllm[:/](\d+\.\d+)", tag)
        if not match:
            self.skipTest(f"Cannot parse vllm version from tag: {tag}")
        expected = match.group(1)
        import vllm

        actual = vllm.__version__
        self.assertTrue(
            actual.startswith(expected),
            f"vllm {actual} doesn't match tag version {expected}",
        )

    def test_python_version_matches_tag(self):
        """Python major.minor should match the image tag."""
        tag = self._parse_image_tag()
        match = re.search(r"py(\d)(\d+)", tag)
        if not match:
            self.skipTest(f"Cannot parse Python version from tag: {tag}")
        expected = f"{match.group(1)}.{match.group(2)}"
        actual = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.assertEqual(actual, expected, f"Python {actual} doesn't match tag version {expected}")

    def test_cuda_version_matches_tag(self):
        """CUDA toolkit major.minor should match the image tag."""
        tag = self._parse_image_tag()
        # Parse "cu129" -> "12.9"
        match = re.search(r"cu(\d+)", tag)
        if not match:
            self.skipTest(f"Cannot parse CUDA version from tag: {tag}")
        cu_digits = match.group(1)  # e.g. "129"
        expected = f"{cu_digits[:-1]}.{cu_digits[-1]}"  # "12.9"
        # Get actual from nvcc
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
        ver_match = re.search(r"release (\d+\.\d+)", result.stdout)
        if not ver_match:
            self.skipTest("Cannot parse nvcc version output")
        actual = ver_match.group(1)
        self.assertEqual(actual, expected, f"CUDA {actual} doesn't match tag version {expected}")

    def test_torch_cuda_matches_toolkit(self):
        """torch.version.cuda should agree with the installed CUDA toolkit."""
        import torch

        torch_cuda = torch.version.cuda  # e.g. "12.9"
        if not torch_cuda:
            self.skipTest("torch not built with CUDA")
        # Get toolkit version from nvcc
        for cuda_home in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda"]:
            nvcc = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.isfile(nvcc):
                result = subprocess.run(
                    [nvcc, "--version"], capture_output=True, text=True, timeout=10
                )
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    toolkit_ver = match.group(1)
                    torch_mm = ".".join(torch_cuda.split(".")[:2])
                    self.assertEqual(
                        torch_mm,
                        toolkit_ver,
                        f"torch CUDA {torch_cuda} vs toolkit {toolkit_ver}",
                    )
                    return
        self.skipTest("Cannot find nvcc to determine toolkit version")

    def test_no_duplicate_packages(self):
        """pip list should not contain duplicate package entries."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            self.skipTest("pip list failed")
        packages = json.loads(result.stdout)
        names = [p["name"].lower() for p in packages]
        dupes = [n for n in set(names) if names.count(n) > 1]
        self.assertEqual(dupes, [], f"Duplicate packages found: {dupes}")


class TestEntrypointContract(unittest.TestCase):
    """Category 4: Verify entrypoint structural correctness."""

    SAGEMAKER_ENTRYPOINTS = ["/usr/local/bin/sagemaker_entrypoint.sh", "/usr/bin/serve"]
    EC2_ENTRYPOINT = "/usr/local/bin/dockerd_entrypoint.sh"

    def _find_sagemaker_entrypoint(self):
        for path in self.SAGEMAKER_ENTRYPOINTS:
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                if "SM_VLLM_" in content or "SM_SGLANG_" in content:
                    return path
        return None

    def test_sagemaker_entrypoint_exists_and_executable(self):
        """SageMaker entrypoint must exist and be executable."""
        ep = self._find_sagemaker_entrypoint()
        if not ep:
            self.skipTest("Not a SageMaker image")
        self.assertTrue(os.access(ep, os.X_OK), f"{ep} is not executable")

    def test_sagemaker_entrypoint_invokes_server(self):
        """SageMaker entrypoint must invoke the correct server module."""
        ep = self._find_sagemaker_entrypoint()
        if not ep:
            self.skipTest("Not a SageMaker image")
        with open(ep) as f:
            content = f.read()
        has_vllm = "vllm.entrypoints.openai.api_server" in content
        has_sglang = "sglang.launch_server" in content
        self.assertTrue(has_vllm or has_sglang, "Entrypoint does not invoke vllm or sglang server")

    def test_sagemaker_entrypoint_default_port_8080(self):
        """SageMaker entrypoint must default to port 8080."""
        ep = self._find_sagemaker_entrypoint()
        if not ep:
            self.skipTest("Not a SageMaker image")
        with open(ep) as f:
            content = f.read()
        self.assertIn("8080", content, "Default port 8080 not found in entrypoint")

    def test_ec2_entrypoint_exists_and_executable(self):
        """EC2 entrypoint must exist and be executable (if present)."""
        if not os.path.isfile(self.EC2_ENTRYPOINT):
            self.skipTest("Not an EC2 image")
        self.assertTrue(
            os.access(self.EC2_ENTRYPOINT, os.X_OK),
            f"{self.EC2_ENTRYPOINT} is not executable",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
