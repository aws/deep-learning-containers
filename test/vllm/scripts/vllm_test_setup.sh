#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# lightning is quarantined on PyPI — remove terratorch (which depends on it)
# from test requirements before resolution. Upstream fixed in vllm#41376.
find vllm_source/requirements -name "*.in" -exec sed -i -E '/(terratorch|lightning)/Id' {} +

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
uv pip compile "${TEST_IN}" -o "${TEST_TXT}" --index-strategy unsafe-best-match --torch-backend cu128 --python-platform x86_64-manylinux_2_28 --python-version 3.12 --prerelease=if-necessary
# dev.txt may live under requirements/ or requirements/dev/ in newer vLLM
if [ -f vllm_source/requirements/dev.txt ]; then
  uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt --torch-backend=auto 2>/dev/null || uv pip install $UV_FLAGS -r "${TEST_TXT}" --torch-backend=auto
elif [ -f vllm_source/requirements/dev/cuda.txt ]; then
  uv pip install $UV_FLAGS -r vllm_source/requirements/dev/cuda.txt --torch-backend=auto 2>/dev/null || uv pip install $UV_FLAGS -r "${TEST_TXT}" --torch-backend=auto
else
  # fallback: install test requirements directly
  uv pip install $UV_FLAGS -r "${TEST_TXT}" --torch-backend=auto
fi
uv pip install $UV_FLAGS pytest pytest-asyncio
# vllm_test_utils may be a package or may not exist in newer versions
if [ -d vllm_source/tests/vllm_test_utils ]; then
  uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
fi
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
# vLLM ≥0.27 already uses src/vllm layout; only move if old layout detected
if [ -d "vllm" ] && [ ! -d "src/vllm" ]; then
  mkdir -p src
  mv vllm src/vllm
fi
# Ensure vllm_source is importable for tests (editable install if pyproject.toml exists)
if [ -f "pyproject.toml" ]; then
  uv pip install $UV_FLAGS -e . --no-build-isolation --no-deps --torch-backend=auto 2>/dev/null || true
fi
