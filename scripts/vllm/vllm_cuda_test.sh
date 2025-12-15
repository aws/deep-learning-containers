#!/bin/bash
set -eux

nvidia-smi
# Platform Tests (CUDA) # 4min
cd vllm_source/tests
pytest -v -s cuda/test_cuda_context.py
