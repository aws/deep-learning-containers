#!/usr/bin/env python3
"""
Sanity tests for the RayTrain DLC (Ray distributed training on the PyTorch
multi-node stack: EFA/NCCL/GDRCopy/OpenMPI + Ray Train/Tune/Data).

Run inside the container:
    docker run --rm --entrypoint python3 <image> /workdir/test/sanity/scripts/test_sanity_ray_train.py

With test gating (env vars from image config, passed by CI):
    docker exec -e EXPECTED_FRAMEWORK_VERSION=2.56.0 \
                -e EXPECTED_DEVICE=gpu \
                <container> python3 /workdir/test/sanity/scripts/test_sanity_ray_train.py

RayTrain is its own sanity script (not folded into test_sanity_training.py)
because it diverges from the PyTorch/TensorFlow training contract in two ways:
it uses EFA's bundled OpenMPI as-is (no from-source double-wrap) and a
passive / KubeRay entrypoint (no /usr/local/bin/entrypoint.sh). It shares the
rest of the training-cluster contract (SSH+MPI+EFA/NCCL+CUDA+venv), which is
verified here directly. This mirrors how the Ray Serve image keeps its own
test suite rather than reusing an inference framework's.

Gating env vars:
    EXPECTED_FRAMEWORK_VERSION - expected Ray version (e.g. "2.56.0")
    EXPECTED_DEVICE            - cpu | gpu (RayTrain is gpu-only today)
"""

import ctypes
import glob
import os
import shutil
import unittest

DEVICE = os.environ.get("EXPECTED_DEVICE", "gpu").lower()
FRAMEWORK_VERSION = os.environ.get("EXPECTED_FRAMEWORK_VERSION", "")

gpu_only = unittest.skipIf(DEVICE != "gpu", "GPU-only test")


class TestContainerEnv(unittest.TestCase):
    """Container-level env vars set in the Dockerfile."""

    def test_dlc_container_type(self):
        self.assertEqual(os.environ.get("DLC_CONTAINER_TYPE"), "training")

    def test_pythondontwritebytecode(self):
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), "1")

    def test_pythonunbuffered(self):
        self.assertEqual(os.environ.get("PYTHONUNBUFFERED"), "1")

    def test_lang(self):
        self.assertEqual(os.environ.get("LANG"), "C.UTF-8")

    def test_efa_nccl_runtime_env(self):
        """EFA/NCCL runtime defaults baked in for the KubeRay/EC2 workflow."""
        self.assertEqual(os.environ.get("FI_PROVIDER"), "efa")
        self.assertEqual(os.environ.get("NCCL_SOCKET_IFNAME"), "eth0")


class TestPath(unittest.TestCase):
    """PATH / LD_LIBRARY_PATH entries. GPU entries gated on EXPECTED_DEVICE."""

    def test_path_venv_bin(self):
        self.assertIn("/opt/venv/bin", os.environ["PATH"])

    def test_path_openmpi(self):
        self.assertIn("/opt/amazon/openmpi/bin", os.environ["PATH"])

    @gpu_only
    def test_path_efa(self):
        self.assertIn("/opt/amazon/efa/bin", os.environ["PATH"])

    @gpu_only
    def test_path_cuda(self):
        self.assertIn("/usr/local/cuda/bin", os.environ["PATH"])

    def test_ld_library_path_openmpi(self):
        self.assertIn("/opt/amazon/openmpi/lib", os.environ.get("LD_LIBRARY_PATH", ""))

    @gpu_only
    def test_ld_library_path_ofi_nccl(self):
        self.assertIn("/opt/amazon/ofi-nccl/lib64", os.environ.get("LD_LIBRARY_PATH", ""))

    @gpu_only
    def test_ld_library_path_cuda(self):
        self.assertIn("/usr/local/cuda/lib64", os.environ.get("LD_LIBRARY_PATH", ""))


class TestBinaries(unittest.TestCase):
    """Binaries on PATH."""

    def test_python_on_path(self):
        self.assertIsNotNone(shutil.which("python"), "python not found on PATH")

    def test_mpirun_on_path(self):
        self.assertIsNotNone(shutil.which("mpirun"), "mpirun not found on PATH")

    def test_ray_on_path(self):
        self.assertIsNotNone(shutil.which("ray"), "ray CLI not found on PATH")

    def test_kuberay_probe_binaries_on_path(self):
        """KubeRay's default liveness/readiness probes shell out to `wget ... | grep`
        against the raylet/gcs healthz endpoints. Both must be present or the Ray
        head/worker pods CrashLoopBackOff on an unmodified KubeRay deployment."""
        self.assertIsNotNone(
            shutil.which("wget"), "wget not found — KubeRay health probe would fail"
        )
        self.assertIsNotNone(
            shutil.which("grep"), "grep not found — KubeRay health probe would fail"
        )

    @gpu_only
    def test_fi_info_on_path(self):
        self.assertIsNotNone(shutil.which("fi_info"), "fi_info not found on PATH")

    @gpu_only
    def test_nvcc_on_path(self):
        """nvcc must be present — runtime-base installs cuda-nvcc so nccl-tests builds."""
        self.assertIsNotNone(shutil.which("nvcc"), "nvcc not found on PATH")


class TestSSHConfig(unittest.TestCase):
    """SSH configuration (shared with the PyTorch training stack, port 22)."""

    def test_sshd_binary(self):
        self.assertTrue(os.access("/usr/sbin/sshd", os.X_OK))

    def test_root_authorized_keys(self):
        self.assertTrue(os.path.isfile("/root/.ssh/authorized_keys"))

    def test_strict_host_key_checking_disabled(self):
        with open("/root/.ssh/config") as f:
            self.assertIn("StrictHostKeyChecking no", f.read())


class TestEFAAndNCCL(unittest.TestCase):
    """EFA + NCCL setup — GPU images only."""

    @gpu_only
    def test_efa_binary_exists(self):
        self.assertTrue(os.access("/opt/amazon/efa/bin/fi_info", os.X_OK))

    @gpu_only
    def test_nccl_config(self):
        with open("/etc/nccl.conf") as f:
            self.assertIn("NCCL_DEBUG=INFO", f.read())

    @gpu_only
    def test_ofi_nccl_plugin_exists(self):
        self.assertTrue(os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net.so"))

    @gpu_only
    def test_all_reduce_perf_exists(self):
        """nccl-tests binary built in runtime-base — used for EFA/NCCL validation."""
        self.assertTrue(os.access("/usr/local/bin/all_reduce_perf", os.X_OK))


class TestCudaRuntime(unittest.TestCase):
    """CUDA runtime — GPU images only."""

    @gpu_only
    def test_cudart_loadable(self):
        ctypes.CDLL("libcudart.so")  # raises OSError if missing

    @gpu_only
    def test_cudart_in_cuda_lib64(self):
        self.assertTrue(
            glob.glob("/usr/local/cuda/lib64/libcudart.so*"),
            "no libcudart.so* in /usr/local/cuda/lib64",
        )


class TestOSSLicenseFiles(unittest.TestCase):
    """OSS license artifacts."""

    def test_python_packages_licenses_exists(self):
        self.assertTrue(os.path.isfile("/root/PYTHON_PACKAGES_LICENSES"))

    def test_third_party_source_code_urls_exists(self):
        self.assertTrue(os.path.isfile("/root/THIRD_PARTY_SOURCE_CODE_URLS"))


class TestVenv(unittest.TestCase):
    """Python venv is at /opt/venv."""

    def test_venv_bin_exists(self):
        self.assertTrue(os.path.isdir("/opt/venv/bin"))


class TestRayTrainStack(unittest.TestCase):
    """RayTrain-specific contract: Ray training stack present, training-scoped."""

    def test_ray_train_stack_imports(self):
        import accelerate  # noqa: F401
        import datasets  # noqa: F401
        import pytorch_lightning  # noqa: F401
        import ray.data  # noqa: F401
        import ray.train  # noqa: F401
        import ray.tune  # noqa: F401
        import transformers  # noqa: F401

    def test_ray_version_matches_expected(self):
        if not FRAMEWORK_VERSION:
            self.skipTest("EXPECTED_FRAMEWORK_VERSION not set")
        import ray

        self.assertEqual(ray.__version__, FRAMEWORK_VERSION)

    def test_torch_is_cuda_build(self):
        import torch

        self.assertIn("cu", torch.__version__)

    def test_ray_default_extra_present(self):
        """ray[default] ships the dashboard + job-submission server that KubeRay's
        health probes and `ray job submit` depend on. Without it the head runs in
        'minimal mode' and job submission fails. aiohttp_cors is a default-only
        dependency, and JobSubmissionClient is the API `ray job submit` uses."""
        import aiohttp_cors  # noqa: F401  (present only with ray[default])
        from ray.job_submission import JobSubmissionClient  # noqa: F401

    def test_ray_serve_extra_absent(self):
        """RayTrain is training-scoped. The `ray.serve` module always ships in the
        ray package, but without the `serve` extra its deps (e.g. starlette) are
        missing and the import fails — a successful import means serve leaked in."""
        with self.assertRaises(ImportError):
            import ray.serve  # noqa: F401


if __name__ == "__main__":
    unittest.main(verbosity=2)
