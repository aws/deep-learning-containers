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

# Regenerate the test lockfile under two layers of constraints.
#
# We have to recompile at all only because the sed above mutates the .in files, which
# makes the checked-in .txt stale. But installing test deps must not mutate the runtime
# the image ships -- the engine is what we are testing. Two failures proved both
# directions of that:
#
#   * Floating free from the unpinned .in pulled apache-tvm-ffi 0.1.13.post2 over the
#     image's 0.1.11 (it is not named in cuda.in; it arrives transitively via xgrammar).
#     Two tvm-ffi .so versions then double-registered the same TypeAttr and aborted the
#     engine in C++: "TypeAttr __ffi_repr__ is already registered for type index 132".
#   * Constraining purely to upstream's .txt then DOWNGRADED opencv-python-headless from
#     the image's 5.0.0.93 to upstream's pinned 4.13.0.90, which links libxcb.so.1 -- a
#     library this image does not ship. cv2 import then failed inside every spawned
#     engine-core worker: "ImportError: libxcb.so.1: cannot open shared object file".
#
# So the image's own installed versions take precedence, and upstream's pins fill in
# everything the image does not already have. Ordering matters: two conflicting pins for
# one package make the resolve unsatisfiable, so upstream lines for packages present in
# the image are dropped rather than layered. If a test dep genuinely cannot satisfy an
# image pin, uv fails loudly here -- which is the right outcome, since silently swapping
# a runtime library out from under the engine is the bug both cases above describe.
TEST_CONSTRAINTS="$(mktemp)"
IMAGE_FREEZE="$(mktemp)"
uv pip freeze $UV_FLAGS 2>/dev/null | grep -E '^[A-Za-z0-9._-]+==' > "${IMAGE_FREEZE}" || true

# Canonical PEP 503 name (lowercase, runs of -_. collapsed to -) so that e.g.
# "opencv_python_headless" and "opencv-python-headless" compare equal.
canon() { tr 'A-Z' 'a-z' | sed -E 's/[-_.]+/-/g'; }
IMAGE_NAMES="$(mktemp)"
sed -E 's/==.*//' "${IMAGE_FREEZE}" | canon | sort -u > "${IMAGE_NAMES}"

cat "${IMAGE_FREEZE}" > "${TEST_CONSTRAINTS}"
if [ -f "${TEST_TXT}" ]; then
  # Keep upstream's pins only for packages the image does not already ship.
  grep -E '^[A-Za-z0-9._-]+==' "${TEST_TXT}" \
    | sed -E '/(terratorch|lightning)/Id' \
    | while IFS= read -r line; do
        name="$(printf '%s' "${line}" | sed -E 's/==.*//' | canon)"
        grep -qxF "${name}" "${IMAGE_NAMES}" || printf '%s\n' "${line}"
      done >> "${TEST_CONSTRAINTS}"
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
