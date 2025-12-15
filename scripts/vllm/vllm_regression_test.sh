#!/bin/bash
set -eux
nvidia-smi

# Regression Test # 7min
cd vllm_source/tests
uv pip install --system modelscope
pytest -v -s test_regression.py
