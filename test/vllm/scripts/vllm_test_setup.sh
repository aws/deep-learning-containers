#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# Upstream PR #39024 (merged Apr 2026) moved requirements/{build,test}.{in,txt}
# into requirements/{build,test}/{cuda,rocm,cpu,xpu}.{in,txt}. Pick whichever
# layout the checked-out vllm_source has.
if [ -f vllm_source/requirements/test/cuda.in ]; then
  TEST_IN="vllm_source/requirements/test/cuda.in"
  TEST_TXT="vllm_source/requirements/test/cuda.txt"
else
  TEST_IN="vllm_source/requirements/test.in"
  TEST_TXT="vllm_source/requirements/test.txt"
fi

# delete old test dependencies file and regen
rm -f "${TEST_TXT}"
# terratorch requires 'lightning' which is quarantined on PyPI — exclude both
sed '/^terratorch/Id' "${TEST_IN}" > /tmp/filtered_test.in
uv pip compile /tmp/filtered_test.in -o "${TEST_TXT}" --index-strategy unsafe-best-match --torch-backend cu130 --python-platform x86_64-manylinux_2_28 --python-version 3.12 --prerelease=if-necessary
sed -i -E '/^(terratorch|lightning)/Id' "${TEST_TXT}"
# uv pip install $UV_FLAGS -r vllm_source/requirements/common.txt --torch-backend=auto
# Filter out terratorch/lightning (lightning is quarantined on PyPI)
sed -E '/^(terratorch|lightning)/Id' vllm_source/requirements/dev.txt > /tmp/filtered_dev.txt
uv pip install $UV_FLAGS -r /tmp/filtered_dev.txt --torch-backend=auto
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm
