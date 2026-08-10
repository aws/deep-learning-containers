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

# Regenerate the test lockfile, but CONSTRAINED by the one upstream shipped.
#
# We have to recompile at all only because the sed above mutates the .in files, which
# makes the checked-in .txt stale. Recompiling from the unpinned .in alone, however,
# throws away every pin upstream chose and floats transitive deps to latest. That is
# not hypothetical: apache-tvm-ffi is not named in cuda.in and arrives transitively via
# xgrammar. Upstream pins it to 0.1.11 -- exactly what the image ships -- but a free
# resolve picked up 0.1.13.post2, which installed over the image's copy and left two
# tvm-ffi .so versions double-registering the same TypeAttr, aborting the engine in C++
# ("TypeAttr __ffi_repr__ is already registered for type index 132") after init.
#
# Using upstream's own .txt as a constraint keeps their pins authoritative while still
# letting the terratorch/lightning removal take effect. Constraints on packages that no
# longer resolve are simply unused, so dropping those two lines from the copy is enough.
TEST_CONSTRAINTS="$(mktemp)"
if [ -f "${TEST_TXT}" ]; then
  sed -E '/(terratorch|lightning)/Id' "${TEST_TXT}" > "${TEST_CONSTRAINTS}"
fi
rm -f "${TEST_TXT}"
uv pip compile "${TEST_IN}" -o "${TEST_TXT}" --constraint "${TEST_CONSTRAINTS}" --index-strategy unsafe-best-match --torch-backend cu130 --python-platform x86_64-manylinux_2_28 --python-version 3.12 --prerelease=if-necessary
uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt --torch-backend=auto
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm
