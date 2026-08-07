#!/usr/bin/env python3
"""Sanity tests for TensorFlow inference DLC images.

Runs inside the container via `docker exec ... python3 test_sanity_tf_inference.py`,
gated by EXPECTED_FRAMEWORK / EXPECTED_DEVICE / EXPECTED_CUSTOMER env vars.
Fast, deterministic, no AWS — catches breakage at sanity stage rather than at
the ~10-min SageMaker endpoint-deploy stage.
"""

import ctypes
import glob
import os
import shutil
import subprocess
import unittest

DEVICE = os.environ.get("EXPECTED_DEVICE", "").lower()
CUSTOMER = os.environ.get("EXPECTED_CUSTOMER", "").lower()
FRAMEWORK = os.environ.get("EXPECTED_FRAMEWORK", "").lower()

gpu_only = unittest.skipIf(DEVICE != "gpu", "GPU-only test")
sagemaker_only = unittest.skipIf(CUSTOMER != "sagemaker", "SageMaker-only test")


class TestContainerEnv(unittest.TestCase):
    """Container-level env vars set in Dockerfile.{cpu,cuda}."""

    def test_dlc_container_type(self):
        self.assertEqual(os.environ.get("DLC_CONTAINER_TYPE"), "inference")

    def test_pythondontwritebytecode(self):
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), "1")

    def test_pythonunbuffered(self):
        self.assertEqual(os.environ.get("PYTHONUNBUFFERED"), "1")

    def test_lang(self):
        self.assertEqual(os.environ.get("LANG"), "C.UTF-8")

    def test_model_base_path(self):
        self.assertEqual(os.environ.get("MODEL_BASE_PATH"), "/models")

    def test_model_name(self):
        self.assertEqual(os.environ.get("MODEL_NAME"), "model")


class TestPath(unittest.TestCase):
    """PATH / LD_LIBRARY_PATH entries required at request time."""

    def test_path_venv_bin(self):
        self.assertIn("/opt/venv/bin", os.environ["PATH"])

    @sagemaker_only
    def test_path_sagemaker(self):
        self.assertIn("/sagemaker", os.environ["PATH"])

    @gpu_only
    def test_path_cuda(self):
        self.assertIn("/usr/local/cuda/bin", os.environ["PATH"])

    @gpu_only
    def test_ld_library_path_cuda(self):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/usr/local/cuda/lib64", ld)


class TestTFServingBinary(unittest.TestCase):
    """tensorflow_model_server binary — the customer request path."""

    BIN = "/usr/local/bin/tensorflow_model_server"

    def test_tfs_present(self):
        self.assertIsNotNone(
            shutil.which("tensorflow_model_server"),
            "tensorflow_model_server not on PATH",
        )

    def test_tfs_executable(self):
        self.assertTrue(os.access(self.BIN, os.X_OK), f"{self.BIN} not executable")

    def test_tfs_version(self):
        """--version resolves + matches EXPECTED_TFS_VERSION."""
        out = subprocess.run([self.BIN, "--version"], capture_output=True, text=True, check=True)
        combined = out.stdout + out.stderr
        self.assertIn("TensorFlow ModelServer", combined)

        expected = os.environ.get("EXPECTED_TFS_VERSION")
        if expected:
            self.assertIn(
                expected,
                combined,
                f"tensorflow_model_server --version should include {expected!r}; got: {combined!r}",
            )

    def test_tfs_shared_libs_resolve(self):
        """No `not found` in ldd — catches missing cuDNN before endpoint deploy."""
        out = subprocess.run(["ldd", self.BIN], capture_output=True, text=True, check=True)
        missing = [line for line in out.stdout.splitlines() if "not found" in line]
        self.assertFalse(
            missing,
            "unresolved shared libraries in tensorflow_model_server:\n" + "\n".join(missing),
        )


@sagemaker_only
class TestHandlerFiles(unittest.TestCase):
    """SageMaker handler artifacts at /sagemaker/."""

    HANDLER_FILES = [
        "/sagemaker/serve",
        "/sagemaker/serve.py",
        "/sagemaker/python_service.py",
        "/sagemaker/tfs_utils.py",
        "/sagemaker/multi_model_utils.py",
        "/sagemaker/tensorflowServing.js",
        "/sagemaker/nginx.conf.template",
    ]

    def test_handler_files_present(self):
        for path in self.HANDLER_FILES:
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(path), f"missing handler file: {path}")


class TestHandlerImports(unittest.TestCase):
    """Python modules imported at request time by python_service.py /
    serve.py. gevent must be first (gevent.monkey.patch_all)."""

    def test_gevent_import(self):
        import gevent  # noqa: F401

    def test_falcon_import(self):
        import falcon  # noqa: F401

    def test_gunicorn_import(self):
        import gunicorn  # noqa: F401

    def test_grpc_import(self):
        import grpc  # noqa: F401

    def test_boto3_import(self):
        import boto3  # noqa: F401

    def test_requests_import(self):
        import requests  # noqa: F401


class TestEntrypoints(unittest.TestCase):
    """SageMaker entrypoint scripts wired via ENTRYPOINT/CMD."""

    ENTRYPOINTS = [
        "/usr/local/bin/dockerd_entrypoint.sh",
        "/usr/local/bin/tf_serving_entrypoint.sh",
    ]

    # SIGTERM-critical scripts: each must exec its payload so the payload
    # becomes PID 1 and receives SIGTERM directly from docker. A regression
    # that drops the exec (leaving bash as PID 1) silently converts every
    # scale-in into a 10s-grace SIGKILL with in-flight requests dropped
    # instead of drained.
    EXEC_REQUIRED_SCRIPTS = [
        "/usr/local/bin/dockerd_entrypoint.sh",
        "/usr/local/bin/tf_serving_entrypoint.sh",
        "/sagemaker/serve",
    ]

    def test_entrypoints_executable(self):
        for path in self.ENTRYPOINTS:
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(path), f"missing: {path}")
                self.assertTrue(os.access(path, os.X_OK), f"not executable: {path}")

    def test_entrypoints_have_shebang(self):
        for path in self.ENTRYPOINTS:
            with self.subTest(path=path):
                with open(path) as f:
                    self.assertTrue(f.readline().startswith("#!"), f"missing shebang: {path}")

    def test_scripts_use_exec_to_hand_off_pid1(self):
        """SIGTERM propagation requires exec (not eval or plain invocation)
        so the payload becomes PID 1. Any of the following would drop signal
        delivery to TFS / serve.py and turn scale-in into SIGKILL after 10s:
            - dropping the ``exec`` keyword ("$@" as a bare command)
            - replacing ``exec`` with ``eval``
            - wrapping in a subshell (``bash -c "..."``)
        """
        for path in self.EXEC_REQUIRED_SCRIPTS:
            with self.subTest(path=path):
                if not os.path.isfile(path):
                    if path == "/sagemaker/serve":
                        # Only present on SageMaker images.
                        continue
                    self.fail(f"missing script: {path}")
                with open(path) as f:
                    content = f.read()
                self.assertNotIn(
                    "eval ",
                    content,
                    f"{path} uses eval — SIGTERM will not propagate to PID 1 payload",
                )
                self.assertRegex(
                    content,
                    r"(?m)^\s*exec\s+\S",
                    f"{path} must contain an exec line so its payload becomes PID 1; "
                    "without it, bash owns PID 1 and docker SIGKILLs after 10s with "
                    "in-flight requests dropped instead of drained",
                )


class TestNginxNjsModule(unittest.TestCase):
    """nginx.conf.template hardcodes an AL2023-specific absolute path — if
    the base image or nginx-mod-njs RPM moves it, nginx -t fails at start."""

    NJS_MODULE = "/usr/lib64/nginx/modules/ngx_http_js_module.so"

    def test_njs_module_present_at_hardcoded_path(self):
        self.assertTrue(
            os.path.isfile(self.NJS_MODULE),
            f"nginx njs module not at {self.NJS_MODULE} — check nginx-mod-njs "
            "RPM install and nginx.conf.template's load_module directive",
        )


# cuDNN 9 sub-libraries — each dlopen'd by TFS on first Conv/RNN/LSTM.
# ldd only reveals the dispatcher (libcudnn.so.9); missing sub-libs slip past.
# Enumerated to match nvidia-cudnn-cu12==9.24.0.43's wheel manifest; keep in
# sync when bumping the cuDNN pin.
CUDNN_9_REQUIRED_SUBLIBS = [
    "libcudnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9",  # transitive via libcudnn_graph DT_NEEDED
    "libcudnn_ext.so.9",  # dlopen'd by name by libcudnn.so.9 dispatcher (new in 9.24 line)
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
]


@gpu_only
class TestCuDNN(unittest.TestCase):
    """cuDNN presence guard + "only-stub-shipped" defense (see PR #6418).

    Asserts each required sub-library is present on disk AND resolvable via
    ldconfig — a build shipping only the dispatcher stub would pass ldd yet
    fault on first Conv/RNN request.
    """

    # Search /usr/local/cuda/lib64 first (Dockerfile stages here), fall back
    # to the nvidia-cudnn-cu12 site-packages layout.
    _CUDNN_SEARCH_GLOBS = (
        "/usr/local/cuda/lib64/{lib}",
        "/opt/venv/lib64/python*/site-packages/nvidia/cudnn/lib/{lib}",
        "/opt/venv/lib/python*/site-packages/nvidia/cudnn/lib/{lib}",
    )

    def _find_on_disk(self, lib: str) -> list:
        matches: list = []
        for pattern in self._CUDNN_SEARCH_GLOBS:
            matches.extend(glob.glob(pattern.format(lib=lib)))
        return matches

    def test_all_cudnn_sublibs_on_disk(self):
        for lib in CUDNN_9_REQUIRED_SUBLIBS:
            with self.subTest(lib=lib):
                found = self._find_on_disk(lib)
                self.assertTrue(
                    found,
                    f"cuDNN sub-library {lib} not found on disk; searched "
                    f"{[p.format(lib=lib) for p in self._CUDNN_SEARCH_GLOBS]}",
                )

    def test_all_cudnn_sublibs_in_ldconfig(self):
        try:
            out = subprocess.check_output(["ldconfig", "-p"], text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.fail(f"ldconfig -p failed: {e}")
        for lib in CUDNN_9_REQUIRED_SUBLIBS:
            with self.subTest(lib=lib):
                self.assertIn(lib, out, f"cuDNN sub-library {lib} not in ldconfig cache")

    def test_all_cudnn_sublibs_dlopen(self):
        """dlopen each — catches truncated, zero-byte, wrong-arch, ABI-drifted."""
        for lib in CUDNN_9_REQUIRED_SUBLIBS:
            with self.subTest(lib=lib):
                try:
                    ctypes.CDLL(lib)
                except OSError as e:
                    self.fail(f"cuDNN sub-library {lib} failed to dlopen: {e}")


class TestOSSLicenseFiles(unittest.TestCase):
    """OSS-compliance artifacts copied from builder-oss stage."""

    ARTIFACTS = [
        "/root/THIRD_PARTY_SOURCE_CODE_URLS",
        "/root/PYTHON_PACKAGES_LICENSES",
        "/root/LINUX_PACKAGES_LICENSES",
        "/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES",
    ]

    def test_oss_artifacts_present(self):
        for path in self.ARTIFACTS:
            with self.subTest(path=path):
                self.assertTrue(os.path.exists(path), f"missing: {path}")


class TestVenv(unittest.TestCase):
    """Python venv exists (builder-base output)."""

    def test_venv_exists(self):
        self.assertTrue(os.path.isdir("/opt/venv/bin"), "/opt/venv/bin missing")


if __name__ == "__main__":
    unittest.main()
