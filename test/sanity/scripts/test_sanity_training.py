#!/usr/bin/env python3
"""
Sanity tests for training DLC images (PyTorch, TensorFlow, XGBoost).

Run inside the container:
    docker run --rm --entrypoint python3 <image> /workdir/test/sanity/scripts/test_sanity_training.py

With test gating (env vars from image config, passed by CI):
    docker exec -e EXPECTED_FRAMEWORK=tensorflow \
                -e EXPECTED_DEVICE=gpu \
                -e EXPECTED_CUSTOMER=sagemaker \
                <container> python3 /workdir/test/sanity/scripts/test_sanity_training.py

Test categories:
    1. TestContainerEnv       - DLC_CONTAINER_TYPE, PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED, LANG
    2. TestPath               - PATH / LD_LIBRARY_PATH entries (some GPU-only)
    3. TestBinaries           - python, mpirun, sshd on PATH (fi_info/nvcc GPU-only)
    4. TestSSHConfig          - sshd binary, root authorized_keys, sshd_config
    5. TestSageMakerPaths     - /opt/ml mount points (sagemaker only)
    6. TestOpenMPI            - OpenMPI wrapper / real binary split (training builds OMPI from source)
    7. TestEFAAndNCCL         - EFA binary, NCCL config, OFI NCCL plugin (GPU only)
    8. TestCudaRuntime        - libcudart loadable + on-disk (GPU only)
    9. TestCuDNN              - libcudnn.so.9 loadable + on-disk (GPU only)
   10. TestCPUImageGuard      - /usr/local/cuda must not exist on CPU image
   11. TestOSSLicenseFiles    - /root/*_LICENSES artifacts
   12. TestVenv               - /opt/venv/bin exists
   13. TestEntrypoint         - /usr/local/bin/entrypoint.sh is executable

Gating env vars:
    EXPECTED_FRAMEWORK - pytorch_runtime | tensorflow | xgboost
    EXPECTED_DEVICE    - cpu | gpu
    EXPECTED_CUSTOMER  - ec2 | sagemaker
"""

import ctypes
import glob
import os
import shutil
import subprocess
import unittest

DEVICE = os.environ.get("EXPECTED_DEVICE", "").lower()
CUSTOMER = os.environ.get("EXPECTED_CUSTOMER", "").lower()

gpu_only = unittest.skipIf(DEVICE != "gpu", "GPU-only test")
cpu_only = unittest.skipIf(DEVICE != "cpu", "CPU-only test")
sagemaker_only = unittest.skipIf(CUSTOMER != "sagemaker", "SageMaker-only test")


class TestContainerEnv(unittest.TestCase):
    """Container-level env vars set in Dockerfile (universal across training frameworks)."""

    def test_dlc_container_type(self):
        self.assertEqual(os.environ.get("DLC_CONTAINER_TYPE"), "training")

    def test_pythondontwritebytecode(self):
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), "1")

    def test_pythonunbuffered(self):
        self.assertEqual(os.environ.get("PYTHONUNBUFFERED"), "1")

    def test_lang(self):
        self.assertEqual(os.environ.get("LANG"), "C.UTF-8")


class TestPath(unittest.TestCase):
    """PATH and LD_LIBRARY_PATH contents. GPU entries gated on EXPECTED_DEVICE == 'gpu'."""

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
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/opt/amazon/openmpi/lib", ld)

    def test_ld_library_path_userlocal(self):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/usr/local/lib", ld)

    @gpu_only
    def test_ld_library_path_ofi_nccl(self):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/opt/amazon/ofi-nccl/lib64", ld)

    @gpu_only
    def test_ld_library_path_efa(self):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/opt/amazon/efa/lib", ld)

    @gpu_only
    def test_ld_library_path_cuda(self):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        self.assertIn("/usr/local/cuda/lib64", ld)


class TestBinaries(unittest.TestCase):
    """Binaries on PATH."""

    def test_python_on_path(self):
        self.assertIsNotNone(shutil.which("python"), "python not found on PATH")

    def test_mpirun_on_path(self):
        self.assertIsNotNone(shutil.which("mpirun"), "mpirun not found on PATH")

    def test_sshd_on_path(self):
        self.assertIsNotNone(shutil.which("sshd"), "sshd not found on PATH")

    @gpu_only
    def test_fi_info_on_path(self):
        self.assertIsNotNone(shutil.which("fi_info"), "fi_info not found on PATH")

    @gpu_only
    def test_nvcc_on_path(self):
        """nvcc binary should be on PATH on CUDA images
        (we install cuda-nvcc-${MAJOR_MINOR} via dnf in runtime-base)."""
        self.assertIsNotNone(shutil.which("nvcc"), "nvcc not found on PATH")


class TestSSHConfig(unittest.TestCase):
    """SSH configuration (universal across training frameworks)."""

    def test_sshd_binary(self):
        self.assertTrue(os.access("/usr/sbin/sshd", os.X_OK))

    def test_root_authorized_keys(self):
        self.assertTrue(os.path.isfile("/root/.ssh/authorized_keys"))

    def test_strict_host_key_checking_disabled(self):
        with open("/root/.ssh/config") as f:
            self.assertIn("StrictHostKeyChecking no", f.read())

    def test_sshd_port_22(self):
        """Port 22 must be the effective port (sshd default, not overridden)."""
        with open("/etc/ssh/sshd_config") as f:
            content = f.read()
        self.assertNotIn("Port 2222", content)

    def test_sshd_root_login_not_disabled(self):
        with open("/etc/ssh/sshd_config") as f:
            content = f.read()
        # sshd default is PermitRootLogin prohibit-password; ensure it's not "no"
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("PermitRootLogin"):
                self.assertNotIn(
                    "no", stripped.lower().split()[-1], "Root login must not be disabled"
                )


class TestSageMakerPaths(unittest.TestCase):
    """SageMaker mount points. Gated on EXPECTED_CUSTOMER == 'sagemaker'."""

    @sagemaker_only
    def test_sagemaker_input_data_exists(self):
        self.assertTrue(os.path.isdir("/opt/ml/input/data"))

    @sagemaker_only
    def test_sagemaker_model_exists(self):
        self.assertTrue(os.path.isdir("/opt/ml/model"))

    @sagemaker_only
    def test_sagemaker_output_exists(self):
        self.assertTrue(os.path.isdir("/opt/ml/output"))

    @sagemaker_only
    def test_sagemaker_code_exists(self):
        self.assertTrue(os.path.isdir("/opt/ml/code"))


class TestOpenMPI(unittest.TestCase):
    """OpenMPI wrapper — training frameworks build OMPI from source."""

    def test_openmpi_binary_exists(self):
        self.assertTrue(os.access("/opt/amazon/openmpi/bin/mpirun", os.X_OK))

    def test_openmpi_double_wrap(self):
        """`mpirun` is a wrapper that exec's `mpirun.real --allow-run-as-root`. Verify the
        wrapper-vs-real split is in place — single grep -c match means it's NOT
        double-wrapped (which would happen if EFA's bundled OMPI wasn't wiped before
        the from-source build)."""
        self.assertTrue(os.access("/opt/amazon/openmpi/bin/mpirun.real", os.X_OK))
        out = subprocess.check_output(
            ["grep", "-c", "mpirun.real", "/opt/amazon/openmpi/bin/mpirun"], text=True
        ).strip()
        self.assertEqual(out, "1", f"expected 1 mpirun.real reference in wrapper, got {out}")


class TestEFAAndNCCL(unittest.TestCase):
    """EFA + NCCL setup — GPU images only."""

    @gpu_only
    def test_efa_binary_exists(self):
        self.assertTrue(os.access("/opt/amazon/efa/bin/fi_info", os.X_OK))

    @gpu_only
    def test_nccl_config(self):
        with open("/etc/nccl.conf") as f:
            content = f.read()
        self.assertIn("NCCL_DEBUG=INFO", content)

    @gpu_only
    def test_ofi_nccl_plugin_exists(self):
        self.assertTrue(os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net.so"))

    @gpu_only
    def test_efa_libfabric_provider(self):
        out = subprocess.check_output(["fi_info", "--version"], text=True, stderr=subprocess.STDOUT)
        self.assertIn("libfabric", out.lower())


class TestCudaRuntime(unittest.TestCase):
    """CUDA runtime — GPU images only."""

    @gpu_only
    def test_cudart_loadable(self):
        """CUDA runtime library must be present and loadable.
        Catches missing or broken libcudart linkage."""
        ctypes.CDLL("libcudart.so")  # raises OSError if missing

    @gpu_only
    def test_cudart_in_cuda_lib64(self):
        self.assertTrue(
            glob.glob("/usr/local/cuda/lib64/libcudart.so*"),
            "no libcudart.so* in /usr/local/cuda/lib64",
        )


class TestCuDNN(unittest.TestCase):
    """cuDNN — GPU images only."""

    @gpu_only
    def test_cudnn_loadable(self):
        """cuDNN runtime library must be present and loadable.
        Catches missing or wrong-SOVERSION libcudnn linkage."""
        ctypes.CDLL("libcudnn.so.9")  # raises OSError if missing

    @gpu_only
    def test_cudnn_in_cuda_lib64(self):
        # cuDNN libs are copied from nvidia-cudnn-cu12 pip pkg into /usr/local/cuda/lib64
        self.assertTrue(
            glob.glob("/usr/local/cuda/lib64/libcudnn*.so*"),
            "no libcudnn*.so* in /usr/local/cuda/lib64",
        )


class TestCPUImageGuard(unittest.TestCase):
    """CPU-image-only guards."""

    @cpu_only
    def test_no_cuda_directory_on_cpu_image(self):
        """The CPU image must not contain /usr/local/cuda — guards against
        accidental CUDA leakage from the pytorch-cpu index workaround."""
        self.assertFalse(
            os.path.isdir("/usr/local/cuda"),
            "/usr/local/cuda exists on CPU image — base image leak?",
        )


class TestOSSLicenseFiles(unittest.TestCase):
    """OSS license artifacts (universal)."""

    def test_python_packages_licenses_exists(self):
        self.assertTrue(os.path.isfile("/root/PYTHON_PACKAGES_LICENSES"))

    def test_linux_packages_licenses_exists(self):
        self.assertTrue(os.path.isfile("/root/LINUX_PACKAGES_LICENSES"))

    def test_build_from_source_licences_exists(self):
        self.assertTrue(os.path.isfile("/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES"))

    def test_third_party_source_code_urls_exists(self):
        self.assertTrue(os.path.isfile("/root/THIRD_PARTY_SOURCE_CODE_URLS"))


class TestVenv(unittest.TestCase):
    """Python venv is at /opt/venv (universal)."""

    def test_venv_bin_exists(self):
        self.assertTrue(os.path.isdir("/opt/venv/bin"))


class TestEntrypoint(unittest.TestCase):
    """Entrypoint script is executable (universal)."""

    def test_entrypoint_executable(self):
        self.assertTrue(os.access("/usr/local/bin/entrypoint.sh", os.X_OK))


if __name__ == "__main__":
    unittest.main(verbosity=2)
