#!/usr/bin/env python3
"""
Sanity tests for TensorFlow inference DLC images.

Runs inside the container:
    docker exec -e EXPECTED_FRAMEWORK=tensorflow \
                -e EXPECTED_DEVICE=gpu \
                -e EXPECTED_CUSTOMER=sagemaker \
                <container> python3 /workdir/test/sanity/scripts/test_sanity_tf_inference.py

Fast, deterministic, no AWS. Fails at the sanity stage instead of at the
~10-minute SageMaker-endpoint-deploy stage when the image is broken.

Test categories:
    1. TestContainerEnv        - DLC_CONTAINER_TYPE=inference + universal env
    2. TestPath                - PATH / LD_LIBRARY_PATH inference-side entries
    3. TestTFServingBinary     - tensorflow_model_server present, --version,
                                 ldd resolves (the exact regression path
                                 that produced audit finding B1)
    4. TestHandlerFiles        - SageMaker handler artifacts at /sagemaker/
    5. TestHandlerImports      - falcon / gunicorn / gevent / grpc / boto3 /
                                 requests importable (handler request path)
    6. TestEntrypoints         - /usr/local/bin/*_entrypoint.sh executable
    7. TestNginxNjsModule      - /usr/lib64/nginx/modules/ngx_http_js_module.so
                                 present (AL2023 hardcode in nginx.conf.template)
    8. TestCuDNN               - each cuDNN 9 sub-library findable on disk AND
                                 in ldconfig cache (GPU only) — defends against
                                 B1 (missing libcudnn) AND the "only dispatcher
                                 stub shipped, sub-libraries missing" failure
                                 that let PR #6418's training-canary regression
                                 through the build-time ldd + libcudnn glob
    9. TestOSSLicenseFiles     - /root/*_LICENSES artifacts
   10. TestVenv                - /opt/venv exists

Gating env vars:
    EXPECTED_FRAMEWORK - tensorflow
    EXPECTED_DEVICE    - cpu | gpu
    EXPECTED_CUSTOMER  - sagemaker
"""

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
    """tensorflow_model_server binary — the customer request path.

    Missing / unlinkable cuDNN here produced audit finding B1 (fixed in this
    PR). These tests keep that class of regression out."""

    BIN = "/usr/local/bin/tensorflow_model_server"

    def test_tfs_present(self):
        self.assertIsNotNone(
            shutil.which("tensorflow_model_server"),
            "tensorflow_model_server not on PATH",
        )

    def test_tfs_executable(self):
        self.assertTrue(os.access(self.BIN, os.X_OK), f"{self.BIN} not executable")

    def test_tfs_version(self):
        """--version resolves + prints a version string. Catches broken
        --no-deps installs of tensorflow-serving-api."""
        out = subprocess.run([self.BIN, "--version"], capture_output=True, text=True, check=True)
        combined = out.stdout + out.stderr
        self.assertIn("TensorFlow ModelServer", combined)

    def test_tfs_shared_libs_resolve(self):
        """No `not found` in ldd. This is the check that would have caught
        B1 (missing cuDNN in GPU image) before endpoint deploy."""
        out = subprocess.run(["ldd", self.BIN], capture_output=True, text=True, check=True)
        missing = [line for line in out.stdout.splitlines() if "not found" in line]
        self.assertFalse(
            missing,
            "unresolved shared libraries in tensorflow_model_server:\n" + "\n".join(missing),
        )


@sagemaker_only
class TestHandlerFiles(unittest.TestCase):
    """SageMaker handler artifacts ported from master TF 2.19
    build_artifacts/sagemaker/ to scripts/docker/tensorflow/inference/
    sagemaker/ in this PR. COPY'd into the image at /sagemaker/."""

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


class TestNginxNjsModule(unittest.TestCase):
    """nginx.conf.template hardcodes an AL2023-specific absolute path
    (commit 3e7012d6). If the base image or nginx-mod-njs RPM ever moves
    the module, nginx -t fails at container start and the endpoint never
    reaches InService."""

    NJS_MODULE = "/usr/lib64/nginx/modules/ngx_http_js_module.so"

    def test_njs_module_present_at_hardcoded_path(self):
        self.assertTrue(
            os.path.isfile(self.NJS_MODULE),
            f"nginx njs module not at {self.NJS_MODULE} — check nginx-mod-njs "
            "RPM install and nginx.conf.template's load_module directive",
        )


# cuDNN 9 sub-libraries. Every one is dlopen'd by TF Serving's GPU code
# path at first Conv/RNN/LSTM invocation; ``ldd tensorflow_model_server``
# only surfaces the direct link to the dispatcher (``libcudnn.so.9``) and
# does NOT reveal a missing sub-library. Enumerating them explicitly here
# closes the "only stub present" hole that let PR #6418's training-canary
# cuDNN regression ship past the build-time ldd + libcudnn glob checks.
CUDNN_9_REQUIRED_SUBLIBS = [
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
]


@gpu_only
class TestCuDNN(unittest.TestCase):
    """B1 regression guard + "only-stub-shipped" defense.

    tensorflow_model_server links dynamically to libcudnn.so; the CUDA base
    image ships CUDA but not cuDNN. This PR copies libcudnn*.so from the
    pip-installed nvidia-cudnn-cu12 package into /usr/local/cuda/lib64 in
    Dockerfile.cuda and runs ldconfig.

    A previous training-canary failure (PR #6418) demonstrated that ``ldd``
    on the binary only surfaces the direct link to the dispatcher
    (``libcudnn.so.9``); TF's autotuning path dlopens each sub-library at
    first kernel launch, so a build shipping only the dispatcher stub
    would satisfy ``ldd`` and a bare ``libcudnn*`` glob yet fault at the
    first customer request. Enumerate every required sub-library and
    assert both presence on disk AND resolvability via ldconfig.
    """

    # Search locations for cuDNN sub-libs. The build copies from
    # nvidia-cudnn-cu12's site-packages layout into /usr/local/cuda/lib64,
    # but preserve the site-packages fallback so this test does not tightly
    # couple to the Dockerfile's staging directory choice.
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
