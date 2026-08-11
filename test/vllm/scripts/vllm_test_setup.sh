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

# Regenerate the test lockfile, preferring the versions the image already ships.
#
# We have to recompile at all only because the sed above mutates the .in files, which makes
# the checked-in .txt stale. But installing test deps must not mutate the runtime the image
# ships -- the engine is what we are testing. Two failures proved both directions of that:
#
#   * Floating free from the unpinned .in pulled apache-tvm-ffi 0.1.13.post2 over the
#     image's 0.1.11 (it is not named in cuda.in; it arrives transitively via xgrammar).
#     Two tvm-ffi .so versions then double-registered the same TypeAttr and aborted the
#     engine in C++: "TypeAttr __ffi_repr__ is already registered for type index 132".
#   * Pinning purely to upstream's .txt then DOWNGRADED opencv-python-headless from the
#     image's 5.0.0.93 to upstream's pinned 4.13.0.90, which links libxcb.so.1 -- a library
#     this image does not ship. cv2 import then failed inside every spawned engine-core
#     worker: "ImportError: libxcb.so.1: cannot open shared object file".
#
# The versions are supplied as PREFERENCES, not constraints. uv considers the versions
# pinned in an existing -o output file and will not upgrade them on a subsequent compile
# (we pass no --upgrade / --upgrade-package), but it yields silently when a requirement in
# the graph forbids the preferred version. That is exactly the policy we want -- "keep what
# the image ships unless upstream genuinely requires otherwise" -- and unlike a constraint
# it cannot make the resolve unsatisfiable.
#
# Constraints were tried first and are the wrong tool: a constraint is a hard requirement,
# so it can never loosen another requirement, and every upstream specifier it contradicts
# is a resolve failure rather than a fallback. That produced a run of them -- grpcio==1.78.0
# in cuda.in vs the image's 1.83.0 ("Because you require grpcio==1.78.0 and
# grpcio==1.83.0"), then runai-model-streamer[s3,gcs,azure]==0.15.7 vs 0.16.1 -- and upper
# bounds such as setuptools<81.0.0, datasets<=3.6.0, fastapi<0.137.0, xgrammar<1.0.0 and
# mteb<3 could each fail the same way the moment the image moves past one. Preferences have
# no such edge, so no per-package special-casing is needed and none is done here.
#
# Upstream's own lockfile pins are kept as preferences too, so packages the image does not
# ship still land on the versions upstream tested. Where both name a package the image wins;
# duplicates are resolved explicitly by name rather than relying on file order, since uv
# does not document which of two pins for one package it honors.
PREFERENCES="$(mktemp)"
IMAGE_FREEZE="$(mktemp)"
uv pip freeze $UV_FLAGS 2>/dev/null | grep -E '^[A-Za-z0-9._-]+==' > "${IMAGE_FREEZE}" || true

# Canonical PEP 503 name (lowercase, runs of -_. collapsed to -) so that e.g.
# "opencv_python_headless" and "opencv-python-headless" compare equal.
canon() { tr 'A-Z' 'a-z' | sed -E 's/[-_.]+/-/g'; }
IMAGE_NAMES="$(mktemp)"
sed -E 's/==.*//' "${IMAGE_FREEZE}" | canon | sort -u > "${IMAGE_NAMES}"

if [ -f "${TEST_TXT}" ]; then
  # Upstream pins, for packages the image does not already ship.
  grep -E '^[A-Za-z0-9._-]+==' "${TEST_TXT}" \
    | sed -E '/(terratorch|lightning)/Id' \
    | while IFS= read -r line; do
        name="$(printf '%s' "${line}" | sed -E 's/==.*//' | canon)"
        grep -qxF "${name}" "${IMAGE_NAMES}" || printf '%s\n' "${line}"
      done > "${PREFERENCES}"
fi
cat "${IMAGE_FREEZE}" >> "${PREFERENCES}"

cp "${PREFERENCES}" "${TEST_TXT}"
uv pip compile "${TEST_IN}" -o "${TEST_TXT}" --index-strategy unsafe-best-match --torch-backend cu130 --python-platform x86_64-manylinux_2_28 --python-version 3.12 --prerelease=if-necessary
uv pip install $UV_FLAGS -r vllm_source/requirements/dev.txt --torch-backend=auto
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS -e vllm_source/tests/vllm_test_utils
uv pip install $UV_FLAGS hf_transfer
cd vllm_source
mkdir src
mv vllm src/vllm
