"""Verify key Python packages import successfully."""

import importlib

import pytest

PACKAGES = [
    "ray",
    "ray.serve",
    "ray.serve.llm",
    "vllm",
    "torch",
    "torchvision",
    "transformers",
    "fastapi",
    "uvicorn",
    "httpx",
    "aiohttp",
    "aiohttp_cors",
    "numpy",
    "boto3",
    "yaml",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_import(pkg):
    importlib.import_module(pkg)


def test_build_openai_app_importable():
    from ray.serve.llm import build_openai_app  # noqa: F401


def test_llm_config_importable():
    from ray.serve.llm import LLMConfig  # noqa: F401


def test_vllm_openai_api_server_importable():
    from vllm.entrypoints.openai import api_server  # noqa: F401


def test_ray_dist_jar_removed():
    import os

    import ray

    jars_dir = os.path.join(os.path.dirname(ray.__file__), "jars")
    jars = (
        [f for f in os.listdir(jars_dir) if f.endswith(".jar")] if os.path.isdir(jars_dir) else []
    )
    assert not jars, jars
