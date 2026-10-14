#!/usr/bin/env python3
"""CPU-safe build smoke test for the Miles AL2023 images.

Expected framework versions come from /opt/aws/dlc/SOURCE_MANIFEST, which the
Dockerfile writes from the build args. Third-party package versions are checked
against uv.lock by validate_uv_lock.py.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

SOURCE_MANIFEST = Path("/opt/aws/dlc/SOURCE_MANIFEST")

CPU_SAFE_IMPORTS = (
    "boto3",
    "botocore",
    "torch",
    "ray",
    "redis",
    "runai_model_streamer",
    "s3transfer",
    "sglang",
    "miles",
    "sglang_router",
    "mooncake.structured_object_store",
    "torchcodec",
    "torchtitan",
)

# megatron is a namespace package shared with Megatron-Bridge, so check
# megatron.core to prove Megatron-LM itself is installed.
GPU_RUNTIME_IMPORTS = (
    "megatron.core",
    "transformer_engine",
    "flash_attn",
    "deep_select",
    "mscclpp",
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def read_manifest() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in SOURCE_MANIFEST.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key and value:
            values[key] = value
    return values


def main() -> None:
    manifest = read_manifest()
    require_cuda = os.environ.get("MILES_SMOKE_REQUIRE_CUDA") == "1"
    imported = {}
    for module_name in CPU_SAFE_IMPORTS:
        module = importlib.import_module(module_name)
        imported[module_name] = getattr(module, "__version__", "imported")

    present = {}
    for module_name in GPU_RUNTIME_IMPORTS:
        spec = importlib.util.find_spec(module_name)
        assert spec is not None, f"missing Python module: {module_name}"
        present[module_name] = str(spec.origin)
        if require_cuda:
            module = importlib.import_module(module_name)
            imported[module_name] = getattr(module, "__version__", "imported")

    torch = importlib.import_module("torch")
    if require_cuda:
        assert torch.cuda.is_available(), "strict CUDA smoke test requires a GPU"

    cuda_major_minor = ".".join(manifest["cuda_version"].split(".")[:2])
    assert platform.system() == "Linux"
    assert "amzn" in platform.platform().lower() or os.path.exists("/etc/system-release")
    assert Version(torch.__version__).base_version == manifest["torch_version"]
    assert torch.version.cuda == cuda_major_minor
    assert Version(package_version("sglang-kernel")).local == "cu" + cuda_major_minor.replace(
        ".", ""
    )
    assert package_version("miles") == manifest["framework_version"]
    assert package_version("sglang") == manifest["sglang_version"]
    from sglang.srt.connector.s3 import S3Connector

    assert S3Connector
    assert os.environ["MILES_SUPPORTED_CUDA_ARCHES"] == manifest["torch_cuda_arch_list"].replace(
        ";", ","
    )
    assert shutil.which("sgl-model-gateway")
    assert shutil.which("mooncake_master")
    assert shutil.which("ffmpeg")
    assert shutil.which("all_reduce_perf")
    assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net-ofi.so")
    assert os.environ["MILES_MOONCAKE_AVAILABLE"] == "1"
    # Miles' Mooncake backend uses the structured object store API.
    store = importlib.import_module("mooncake.structured_object_store")
    for name in ("FieldSchema", "export_ref", "import_ref"):
        assert hasattr(store, name), f"mooncake.structured_object_store lacks {name}"
    for name in ("put", "get", "release_result", "cleanup_dataproto"):
        assert hasattr(store.MooncakeBundleTransfer, name), f"MooncakeBundleTransfer lacks {name}"

    report = {
        "imports": imported,
        "present_without_gpu_import": present,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "torchaudio",
                "ray",
                "miles",
                "sglang",
                "sglang-kernel",
                "flashinfer-python",
                "transformer-engine",
                "sglang-router",
                "mooncake-transfer-engine-cuda13",
                "torchcodec",
                "boto3",
                "botocore",
                "s3transfer",
            )
        },
        "cuda": torch.version.cuda,
        "cuda_visible": torch.cuda.is_available(),
        "strict_cuda_imports": require_cuda,
        "supported_cuda_arches": os.environ["MILES_SUPPORTED_CUDA_ARCHES"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
