"""Verify installed package versions match pins from versions.env."""

import importlib.metadata
import os
import re
import sys

import pytest

# Detect GPU vs CPU image by checking for CUDA, then pick the right versions file.
_WORKDIR = os.environ.get("DLC_WORKDIR", "/workdir")
IS_CUDA = os.path.isdir("/usr/local/cuda")
_VERSIONS_FILE = "versions-cuda.env" if IS_CUDA else "versions-cpu.env"
VERSIONS_ENV = os.path.join(_WORKDIR, "docker", "tensorflow", _VERSIONS_FILE)
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")


def _parse_versions_env():
    versions = {}
    with open(VERSIONS_ENV) as f:
        for line in f:
            m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
            if m:
                versions[m.group(1)] = m.group(2)
    return versions


ENV = _parse_versions_env()


def test_tensorflow_version():
    """The CPU image installs the `tensorflow_cpu` distribution, the CUDA image
    installs `tensorflow`. Both expose the version through importlib.metadata
    (and at runtime as `tensorflow.__version__`)."""
    dist_name = "tensorflow" if IS_CUDA else "tensorflow_cpu"
    actual = importlib.metadata.version(dist_name)
    expected = ENV["TF_VERSION"]
    # Compare just X.Y.Z (TF_VERSION is "2.21.0" — no suffix expected).
    assert actual.startswith(expected), f"{dist_name}: expected {expected}*, got {actual}"


def test_python_version():
    expected = ENV["PYTHON_VERSION"]
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert actual == expected, f"Expected Python {expected}, got {actual}"


@cuda_only
def test_cuda_runtime_version_matches_env():
    """Verify the CUDA runtime installed in the image matches versions-cuda.env.

    AL2023 nvidia/cuda images don't ship /usr/local/cuda/version.json. They DO
    ship a versioned directory like /usr/local/cuda-12.6 with /usr/local/cuda
    as a symlink to it. Reading the symlink target gives us the major.minor
    cleanly. Patch level isn't represented in the directory name (NVIDIA's
    convention), so we only compare major.minor."""
    expected = ".".join(ENV["CUDA_VERSION"].split(".")[:2])  # e.g. "12.6"
    target = os.readlink("/usr/local/cuda")  # e.g. "/usr/local/cuda-12.6" or "cuda-12.6"
    # Extract trailing "X.Y" — robust to absolute vs relative symlink targets
    m = re.search(r"cuda-(\d+\.\d+)", target)
    assert m is not None, f"Could not parse CUDA version from symlink target: {target}"
    actual = m.group(1)
    assert actual == expected, (
        f"CUDA runtime mismatch: image has {actual} (symlink {target}), "
        f"versions-cuda.env says {expected}"
    )


@cuda_only
def test_tensorflow_cuda_compile_target_forward_compatible():
    """Verify TF was compiled against a CUDA version forward-compatible with
    our base image runtime.

    NVIDIA's forward minor-version compatibility: code compiled against
    CUDA X.Y can run on any runtime X.Z where Z >= Y, same major X.

    TF 2.21 was compiled against CUDA 12.5; our base is 12.6+. This test
    encodes that locked decision so we catch a future TF wheel that's
    compiled against a CUDA version incompatible with our base image."""
    import tensorflow as tf

    build_info = tf.sysconfig.get_build_info()
    tf_compile_cuda = build_info["cuda_version"]  # e.g. "12.5"
    runtime_cuda = ".".join(ENV["CUDA_VERSION"].split(".")[:2])  # e.g. "12.6"

    tf_major, tf_minor = (int(x) for x in tf_compile_cuda.split(".")[:2])
    rt_major, rt_minor = (int(x) for x in runtime_cuda.split(".")[:2])

    assert tf_major == rt_major, (
        f"CUDA major mismatch: TF compiled against {tf_compile_cuda}, "
        f"runtime is {runtime_cuda}. Forward-compat requires same major."
    )
    assert tf_minor <= rt_minor, (
        f"TF compiled against newer CUDA than runtime: "
        f"TF={tf_compile_cuda}, runtime={runtime_cuda}. "
        f"Forward-compat requires TF compile-time minor <= runtime minor."
    )
