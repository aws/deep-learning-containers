#!/bin/bash
set -eux
nvidia-smi

UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# Regression Test # 7min
cd vllm_source/tests
# modelscope<1.38: 1.38+ removed `revision` kwarg from LegacyHubApi.get_model_files(),
# breaking vLLM's repo_utils.py. Unpin after upstream vLLM adapts to the new API.
# unpin after https://github.com/vllm-project/vllm/pull/47325 is merged
uv pip install $UV_FLAGS "modelscope<1.38"
pytest -v -s test_regression.py
