"""Verify the Miles framework stack is importable without a GPU."""

import csv
import importlib
import importlib.util
import re
from importlib import metadata
from pathlib import Path, PurePosixPath

import pytest

CPU_IMPORTS = [
    "boto3",
    "miles",
    "mooncake.structured_object_store",
    # The output store SGLang imports, from the SGLang checkout.
    "radix_transfer_plugins.stashes.token_output",
    "ray",
    "redis",
    "runai_model_streamer",
    "s3transfer",
    "sglang",
    # SGLang's prebuilt Rust TreeCore, loaded as SGLang loads it. The image drops the rust/
    # sources it was built from, so the loader must take the bundled extension.
    "sglang.srt.mem_cache.rust_tree_core.extension",
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
    "flash_mla",
    "megatron.core",
    "mscclpp",
    "sgl_flash_mla",
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


def package_files(module):
    directory = Path(importlib.util.find_spec(module).submodule_search_locations[0])
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def recorded_files(distribution):
    # Read RECORD itself: metadata.files() drops entries whose path no longer exists, and
    # sgl-flash-mla's record still names the flash_mla/ directory its wheel installs to.
    paths = [
        PurePosixPath(row[0])
        for row in csv.reader(metadata.distribution(distribution).read_text("RECORD").splitlines())
        if row
    ]
    return {
        PurePosixPath(*path.parts[1:]).as_posix()
        for path in paths
        if path.parts[0] == "flash_mla" and "__pycache__" not in path.parts
    }


@pytest.mark.parametrize(
    "module, distribution", [("flash_mla", "flash-mla"), ("sgl_flash_mla", "sgl-flash-mla")]
)
def test_flash_mla_builds_do_not_mix(module, distribution):
    """Two FlashMLA builds ship, as in upstream's image: flash_mla is the nv_dev release
    wheel that Megatron and Miles import, and sgl_flash_mla is SGLang's sgl-flash-mla.
    Each package holds exactly its own build's files."""
    assert package_files(module) == recorded_files(distribution)


def test_sgl_flash_mla_does_not_import_flash_mla():
    """SGLang's FlashMLA build must not load any part of the nv_dev build."""
    directory = Path(importlib.util.find_spec("sgl_flash_mla").submodule_search_locations[0])
    for source in sorted(directory.glob("*.py")):
        assert not re.search(r"^\s*(from|import) flash_mla\b", source.read_text(), re.MULTILINE), (
            f"{source.name} imports flash_mla"
        )
