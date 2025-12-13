#!/bin/bash
set -eux

# delete old test dependencies file and regen
rm vllm_source/requirements/test.txt
uv pip compile vllm_source/requirements/test.in -o vllm_source/requirements/test.txt --index-strategy unsafe-best-match --torch-backend cu129 --python-platform x86_64-manylinux_2_28 --python-version 3.12
uv pip install --system -r vllm_source/requirements/common.txt -r vllm_source/requirements/dev.txt --torch-backend=auto
uv pip install --system pytest pytest-asyncio
uv pip install --system -e vllm_source/tests/vllm_test_utils
uv pip install --system hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm