#!/bin/bash
set -eux

uv pip install --system -r vllm_source/requirements/common.txt -r vllm_source/requirements/dev.txt --torch-backend=auto
uv pip install --system pytest pytest-asyncio
uv pip install --system -e vllm_source/tests/vllm_test_utils
uv pip install --system hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm