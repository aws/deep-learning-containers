#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# lightning is quarantined on PyPI — remove terratorch (which depends on it)
# from test requirements before resolution. Upstream fixed in vllm#41376.
find vllm_source/requirements -name "*.in" -exec sed -i -E '/(terratorch|lightning)/Id' {} + 2>/dev/null || true
find vllm_source/requirements -name "*.txt" -exec sed -i -E '/(terratorch|lightning)/Id' {} + 2>/dev/null || true

# Snapshot base image runtime libs to restore after test deps install; test reqs
# transitively clobber flashinfer<->tvm-ffi (crashes EngineCore) and dual-install cupy.
PROTECTED_PKGS="apache-tvm-ffi cupy-cuda13x cupy-cuda12x flashinfer-python flashinfer-jit-cache"
BASE_FREEZE="$(uv pip freeze $UV_FLAGS 2>/dev/null || true)"

# Upstream PR #39024 (merged Apr 2026) moved requirements/{build,test}.{in,txt}
# into requirements/{build,test}/{cuda,rocm,cpu,xpu}.{in,txt}. Pick whichever
# layout the checked-out vllm_source has.
if [ -f vllm_source/requirements/test/cuda.in ]; then
  TEST_IN="vllm_source/requirements/test/cuda.in"
  TEST_TXT="vllm_source/requirements/test/cuda.txt"
elif [ -f vllm_source/requirements/test.in ]; then
  TEST_IN="vllm_source/requirements/test.in"
  TEST_TXT="vllm_source/requirements/test.txt"
else
  # vLLM 0.27+ may only have a pre-compiled txt without .in
  TEST_IN=""
  TEST_TXT=""
fi

# delete old test dependencies file and regen
if [ -n "${TEST_IN}" ] && [ -f "${TEST_IN}" ]; then
  rm -f "${TEST_TXT}"
  uv pip compile "${TEST_IN}" -o "${TEST_TXT}" --index-strategy unsafe-best-match --torch-backend cu130 --python-platform x86_64-manylinux_2_28 --python-version 3.12 --prerelease=if-necessary
fi
# dev.txt may live at requirements/dev.txt or requirements/dev/cuda.txt
if [ -f vllm_source/requirements/dev.txt ]; then
  uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt --torch-backend=auto
elif [ -f vllm_source/requirements/dev/cuda.txt ]; then
  uv pip install $UV_FLAGS -r vllm_source/requirements/dev/cuda.txt --torch-backend=auto
elif [ -n "${TEST_TXT}" ] && [ -f "${TEST_TXT}" ]; then
  # Fall back to installing test requirements directly
  uv pip install $UV_FLAGS -r "${TEST_TXT}" --torch-backend=auto
else
  echo "WARNING: No dev or test requirements file found, installing minimal test deps"
  uv pip install $UV_FLAGS pytest pytest-asyncio --torch-backend=auto
fi
uv pip install $UV_FLAGS pytest pytest-asyncio
if [ -d "vllm_source/tests/vllm_test_utils" ]; then
  uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
else
  echo "WARNING: vllm_test_utils not found, skipping"
fi
uv pip install $UV_FLAGS hf_transfer

# Restore base image pins for PROTECTED_PKGS if test deps changed them, and drop
# any package the test deps added that the base image didn't have (e.g. cupy-cuda12x).
if [ -n "${BASE_FREEZE}" ]; then
  cur_freeze="$(uv pip freeze $UV_FLAGS 2>/dev/null || true)"
  for pkg in ${PROTECTED_PKGS}; do
    # Match name with either hyphen or underscore separators (freeze may use either)
    pkg_re="^$(printf '%s' "${pkg}" | sed 's/[-_]/[-_]/g')=="
    base_pin="$(printf '%s\n' "${BASE_FREEZE}" | grep -iE "${pkg_re}" || true)"
    cur_pin="$(printf '%s\n' "${cur_freeze}" | grep -iE "${pkg_re}" || true)"
    if [ -z "${base_pin}" ]; then
      # Not in base image; remove if a test dep added it (avoids conflicting stack).
      if [ -n "${cur_pin}" ]; then
        echo "Removing ${pkg} (not present in base image): ${cur_pin}"
        uv pip uninstall $UV_FLAGS "${pkg}" || true
      fi
    elif [ "${base_pin}" != "${cur_pin}" ]; then
      echo "Restoring base image pin for ${pkg}: ${cur_pin:-<none>} -> ${base_pin}"
      uv pip install $UV_FLAGS "${base_pin}"
    fi
  done
fi

cd vllm_source
# vLLM may already use src/ layout; only move if needed
if [ -d "vllm" ] && [ ! -d "src/vllm" ]; then
  mkdir -p src
  mv vllm src/vllm
fi

# vLLM 0.27+ may have moved vllm_test_utils location
if [ ! -d "tests/vllm_test_utils" ]; then
  echo "WARNING: tests/vllm_test_utils not found, skipping editable install"
fi
