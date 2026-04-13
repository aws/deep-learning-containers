#!/bin/bash
set -eux
nvidia-smi

UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# Regression Test # 7min
cd vllm_source/tests
uv pip install $UV_FLAGS modelscope
pytest -v -s test_regression.py
