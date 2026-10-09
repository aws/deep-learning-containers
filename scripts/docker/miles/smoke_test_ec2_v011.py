#!/usr/bin/env python3
"""CPU-safe build smoke test for the Miles v0.1.1 AL2023 EC2 image."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

CPU_SAFE_IMPORTS = (
    "boto3",
    "botocore",
    "torch",
    "ray",
    "s3transfer",
    "sglang",
    "miles",
    "sglang_router",
    "mooncake.structured_object_store",
    "torchcodec",
)

GPU_RUNTIME_IMPORTS = (
    "megatron",
    "transformer_engine",
    "flash_attn",
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
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

    assert platform.system() == "Linux"
    assert "amzn" in platform.platform().lower() or os.path.exists(
        "/etc/system-release"
    )
    assert package_version("torch").startswith("2.13.0")
    assert package_version("ray") == "2.58.0"
    sglang_kernel_version = Version(package_version("sglang-kernel"))
    assert sglang_kernel_version.base_version == "0.4.7"
    assert sglang_kernel_version.local == "cu130"
    assert package_version("flashinfer-python") == "0.6.18"
    assert package_version("miles") == "0.1.1"
    assert package_version("torchcodec") == "0.15.0+cu130"
    assert package_version("boto3") == "1.43.97"
    assert package_version("botocore") == "1.43.97"
    assert package_version("s3transfer") == "0.19.2"
    from sglang.srt.connector.s3 import S3Connector

    assert S3Connector
    assert os.environ["MILES_SUPPORTED_CUDA_ARCHES"] == (
        "8.0,8.6,8.9,9.0,10.0,10.3"
    )
    assert shutil.which("sgl-model-gateway")
    assert shutil.which("mooncake_master")
    assert shutil.which("ffmpeg")
    assert shutil.which("all_reduce_perf")
    assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net-ofi.so")
    assert os.environ["MILES_MOONCAKE_AVAILABLE"] == "1"
    mooncake = importlib.import_module("mooncake")
    mooncake_root = Path(mooncake.__file__).parent
    assert list(mooncake_root.glob("ep_2_13_0*.so"))
    assert list(mooncake_root.glob("pg_2_13_0*.so"))

    if require_cuda:
        importlib.import_module("mooncake.ep")
        importlib.import_module("mooncake.pg")

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
