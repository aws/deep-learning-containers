#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# delete old test dependencies file and regen
rm vllm_source/requirements/test.txt
uv pip compile vllm_source/requirements/test.in -o vllm_source/requirements/test.txt --index-strategy unsafe-best-match --torch-backend cu129 --python-platform x86_64-manylinux_2_28 --python-version 3.12
# uv pip install $UV_FLAGS -r vllm_source/requirements/common.txt --torch-backend=auto
uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm