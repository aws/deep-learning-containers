#!/bin/bash
set -eux
nvidia-smi

UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# Regression Test # 7min
cd vllm_source/tests
# modelscope 1.38.0 dropped get_model_files(revision=), breaks vllm 0.24.0
uv pip install $UV_FLAGS 'modelscope<1.38'
pytest -v -s test_regression.py
