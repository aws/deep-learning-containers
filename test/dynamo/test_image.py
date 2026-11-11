"""Image contract tests for the Dynamo DLC (CPU only, run inside the image).

The workflow passes what the image config declares:
  DYNAMO_BACKEND           metadata.backend (vllm | sglang | trtllm)
  EXPECTED_DYNAMO_VERSION  metadata.framework_version (e.g. 1.6.2)
  EXPECTED_BACKEND_VERSION metadata.backend_version (e.g. 0.31.0)
  EFA_FLAVOR               build.efa_flavor (efa | noefa)
"""

import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

BACKEND = os.environ["DYNAMO_BACKEND"]
# Python distribution of each backend, when it differs from the backend name.
BACKEND_DIST = {"trtllm": "tensorrt-llm"}.get(BACKEND, BACKEND)
EFA_FLAVOR = os.environ["EFA_FLAVOR"]
LIBFABRIC = Path("/opt/amazon/efa/lib64/libfabric.so.1")
LOCK = Path("/opt/dynamo-env/uv.lock")
# NIXL versions whose wheel LIBFABRIC plugin needs a rebuild with an upstream fix
# (scripts/docker/dynamo/nixl/patches/<version>/): ai-dynamo/nixl#2308.
NIXL_PATCHED_VERSIONS = {"1.5.0"}


def base_version(version):
    """Drop the local part (`+dlc1`, `+cu130`) so build labels don't affect the comparison."""
    return version.split("+", 1)[0]


def nixl_distribution():
    """The nixl-cuXX wheel, or None when the backend bundles NIXL (TensorRT-LLM)."""
    for name in ("nixl-cu13", "nixl-cu12"):
        try:
            return md.distribution(name)
        except md.PackageNotFoundError:
            continue
    return None


def canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


def test_environment_matches_lock():
    """Every package in the backend's uv.lock is installed at the locked version, and nothing
    else is (the image is built with `uv sync --frozen`)."""
    # The lock's own project entry (dynamo-dlc-<backend>) is never installed (--no-install-project).
    locked = {
        canonical(p["name"]): p["version"]
        for p in tomllib.loads(LOCK.read_text())["package"]
        if not p["name"].startswith("dynamo-dlc-")
    }
    # Only the image's environment; test tooling (pytest) lives outside it on PYTHONPATH.
    installed = {
        canonical(d.metadata["Name"]): d.version
        for d in md.distributions()
        if str(d.locate_file("")).startswith(sys.prefix)
    }
    # Not in the lock: built from Dynamo source (unpublished).
    installed.pop("gpu-memory-service", None)
    mismatched = {n: (v, installed.get(n)) for n, v in locked.items() if installed.get(n) != v}
    assert not mismatched, f"locked vs installed: {mismatched}"
    extra = sorted(set(installed) - set(locked))
    assert not extra, f"installed but not in uv.lock: {extra}"


def test_environment_dependencies_are_consistent():
    """`uv pip check` passes, except for the lock's declared exclude-dependencies."""
    result = subprocess.run(
        ["python3", "/opt/dynamo-env/check_environment.py", "/opt/dynamo-env/pyproject.toml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dynamo_version_matches_config():
    expected = os.environ["EXPECTED_DYNAMO_VERSION"]
    assert md.version("ai-dynamo") == expected
    assert md.version("ai-dynamo-runtime") == expected


def test_backend_version_matches_config():
    expected = base_version(os.environ["EXPECTED_BACKEND_VERSION"])
    assert base_version(md.version(BACKEND_DIST)) == expected


@pytest.mark.parametrize(
    "module",
    [
        "dynamo.frontend",
        f"dynamo.{BACKEND}",
        "gpu_memory_service.cli.server",
        # TensorRT-LLM bundles NIXL as native libraries, without the nixl Python package; its KV
        # transfer goes through the C++ transfer agent binding (no pure-Python fallback installed).
        *(
            ["nixl._api"]
            if BACKEND != "trtllm"
            else ["tensorrt_llm.tensorrt_llm_transfer_agent_binding"]
        ),
    ],
)
def test_dynamo_modules_import(module):
    # In a subprocess with CUDA's compat libcuda: some backends (TensorRT-LLM) load the driver
    # library at import time, and CPU test runners have no driver.
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, ["/usr/local/cuda/compat", env.get("LD_LIBRARY_PATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_dynamo_source_manifest_present():
    manifest = Path("/opt/dynamo/DYNAMO_SOURCE_MANIFEST")
    assert manifest.is_file()
    assert "HEAD: " in manifest.read_text()


@pytest.mark.skipif(EFA_FLAVOR != "efa", reason="EFA flavor only")
def test_efa_flavor_has_efa_stack():
    assert LIBFABRIC.exists(), "EFA libfabric missing"
    assert Path("/opt/amazon/ofi-nccl").is_dir(), "aws-ofi-nccl plugin missing"
    assert shutil.which("all_reduce_perf"), "nccl-tests all_reduce_perf missing"
    # Built from source, so it must be listed for OSS compliance.
    assert "nccl-tests" in Path("/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES").read_text()


@pytest.mark.skipif(EFA_FLAVOR != "noefa", reason="non-EFA flavor only")
def test_noefa_flavor_has_no_efa_stack():
    assert not LIBFABRIC.exists()
    assert not Path("/opt/amazon/ofi-nccl").exists()


@pytest.mark.skipif(BACKEND != "trtllm", reason="TensorRT-LLM only")
def test_trtllm_nixl_ucx_plugin_loads_after_mpi_init():
    """TensorRT-LLM initializes MPI before its KV transfer agent loads the bundled NIXL UCX plugin. If
    Open MPI has loaded a system UCX (AL2023's openmpi pulls in UCX 1.12) the plugin fails with an
    undefined symbol, and disaggregated serving without EFA breaks."""
    code = """
import ctypes, importlib.util, os
from mpi4py import MPI
MPI.COMM_WORLD.Barrier()
d = os.path.dirname(importlib.util.find_spec("tensorrt_llm").origin)
ctypes.CDLL(os.path.join(d, "libs/nixl/plugins/libplugin_UCX.so"), mode=os.RTLD_NOW)
"""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, ["/usr/local/cuda/compat", env.get("LD_LIBRARY_PATH")])
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.skipif(EFA_FLAVOR != "efa", reason="EFA flavor only")
def test_nixl_libfabric_plugin_ready_for_efa():
    """The backend's NIXL can use EFA:
    - nixl-cu13 1.5.0 races on the EFA endpoint (aborts SGLang's KV transfer over EFA), so its
      LIBFABRIC plugin must be rebuilt with ai-dynamo/nixl#2308;
    - TensorRT-LLM's bundled NIXL ships without LIBFABRIC, so the image must add it."""
    nixl = nixl_distribution()
    patches = Path("/opt/nixl/LIBFABRIC_PLUGIN_PATCHES")
    if nixl is None:
        spec = importlib.util.find_spec("tensorrt_llm")
        assert spec is not None, "no nixl-cuXX wheel and no tensorrt_llm"
        plugin = (
            Path(spec.submodule_search_locations[0]) / "libs/nixl/plugins/libplugin_LIBFABRIC.so"
        )
        assert plugin.is_file(), f"{plugin} missing"
        assert (
            Path("/opt/nixl/LIBFABRIC_PLUGIN_SOURCE")
            .read_text()
            .startswith("tensorrt_llm-bundled-nixl ")
        )
    elif nixl.version in NIXL_PATCHED_VERSIONS:
        assert patches.is_file(), f"{nixl.metadata['Name']} {nixl.version} plugin not rebuilt"
        assert "nixl-2308" in patches.read_text()
    else:
        assert not patches.exists(), f"plugin rebuilt for unexpected NIXL {nixl.version}"
    # A rebuilt plugin is built from source, so it must be listed for OSS compliance.
    rebuilt = Path("/opt/nixl/LIBFABRIC_PLUGIN_SOURCE").exists()
    listed = (
        "NIXL LIBFABRIC plugin" in Path("/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES").read_text()
    )
    assert listed == rebuilt, f"plugin rebuilt: {rebuilt}, listed in BUILD_FROM_SOURCE: {listed}"


@pytest.mark.skipif(EFA_FLAVOR != "efa", reason="EFA flavor only")
def test_patched_nixl_plugin_has_modification_notice():
    """A LIBFABRIC plugin built with upstream fixes carries a modification notice linking each fix
    (Apache-2.0 4(b)) in both OSS attribution files."""
    patches = Path("/opt/nixl/LIBFABRIC_PLUGIN_PATCHES")
    if not patches.is_file():
        pytest.skip("plugin not patched")
    prs = [
        line.rsplit("nixl-", 1)[1].removesuffix(".patch") for line in patches.read_text().split()
    ]
    for name in ("BUILD_FROM_SOURCE_PACKAGES_LICENCES", "THIRD_PARTY_SOURCE_CODE_URLS"):
        entry = next(
            (
                line
                for line in Path("/root", name).read_text().splitlines()
                if line.startswith(("** NIXL LIBFABRIC plugin;", "nixl "))
            ),
            "",
        )
        assert "modified" in entry, f"{name}: no modification notice in {entry!r}"
        for pr in prs:
            assert f"/nixl/pull/{pr}" in entry, f"{name}: fix {pr} not linked in {entry!r}"


def test_python_packages_without_license_file_are_attributed():
    """Wheels that ship no license file (left out by the OSS compliance tool) get their license
    text from scripts/docker/dynamo/add_python_licenses.sh."""
    licenses = Path("/root/PYTHON_PACKAGES_LICENSES").read_text()
    for dist, marker in {"nvshmem4py-cu13": "Apache License"}.items():
        try:
            md.distribution(dist)
        except md.PackageNotFoundError:
            continue
        start = licenses.find(f"Package Name: {dist}\n")
        assert start >= 0, f"{dist} missing from PYTHON_PACKAGES_LICENSES"
        end = licenses.find("\nPackage Name: ", start + 1)
        assert marker in licenses[start : end if end > 0 else None], f"{dist}: no license text"
