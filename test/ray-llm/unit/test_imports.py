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
