"""Verify the Miles framework stack is importable without a GPU."""

import importlib
import importlib.util

import pytest

CPU_IMPORTS = [
    "boto3",
    "miles",
    "mooncake.structured_object_store",
    "ray",
    "redis",
    "runai_model_streamer",
    "s3transfer",
    "sglang",
    "sglang_router",
    "torch",
    "torchcodec",
    "torchtitan",
]

# CUDA extensions that need a GPU to import, so check they are installed. megatron
# is a namespace package shared with Megatron-Bridge; megatron.core is Megatron-LM.
GPU_EXTENSIONS = [
    "deep_select",
    "flash_attn",
    "megatron.core",
    "mscclpp",
    "transformer_engine",
]


@pytest.mark.parametrize("module", CPU_IMPORTS)
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", GPU_EXTENSIONS)
def test_gpu_extension_installed(module):
    assert importlib.util.find_spec(module) is not None, f"{module} is not installed"


def test_mooncake_structured_object_store_api():
    store = importlib.import_module("mooncake.structured_object_store")
    for name in ("FieldSchema", "export_ref", "import_ref"):
        assert hasattr(store, name), f"mooncake.structured_object_store lacks {name}"
    for name in ("put", "get", "release_result", "cleanup_dataproto"):
        assert hasattr(store.MooncakeBundleTransfer, name), f"MooncakeBundleTransfer lacks {name}"


@pytest.mark.parametrize("connector", ["s3.S3Connector", "redis.RedisConnector"])
def test_sglang_model_connector_imports(connector):
    module, _, name = connector.rpartition(".")
    assert getattr(importlib.import_module(f"sglang.srt.connector.{module}"), name)
