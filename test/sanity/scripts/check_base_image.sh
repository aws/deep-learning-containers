#!/usr/bin/env bash
set -euo pipefail

# Base image sanity checks (CPU only, no GPU required)
# Usage: check_base_image.sh <python_version> <cuda_version>
# Example: check_base_image.sh 3.13 12.9

PYTHON_VERSION="${1:?Usage: check_base_image.sh <python_version> <cuda_version>}"
CUDA_VERSION="${2:?Usage: check_base_image.sh <python_version> <cuda_version>}"
FAILED=0

# --- Python version matches expected ---
ACTUAL_PY=$(python3 --version 2>&1 | awk '{print $2}')
ACTUAL_PY_SHORT="${ACTUAL_PY%.*}"
if [ "$ACTUAL_PY_SHORT" = "$PYTHON_VERSION" ]; then
  echo "PASS: Python version $ACTUAL_PY (expected $PYTHON_VERSION.x)"
else
  echo "FAIL: Python version $ACTUAL_PY does not match expected $PYTHON_VERSION"
  FAILED=1
fi

# --- pip is functional ---
if pip --version >/dev/null 2>&1; then
  echo "PASS: pip is functional"
else
  echo "FAIL: pip --version failed"
  FAILED=1
fi

# --- Secure compiler flags: stack protector ---
CFLAGS=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('CFLAGS'))" 2>/dev/null)
if echo "$CFLAGS" | grep -q -- "-fstack-protector-strong"; then
  echo "PASS: -fstack-protector-strong in CFLAGS"
else
  echo "FAIL: -fstack-protector-strong missing from CFLAGS: $CFLAGS"
  FAILED=1
fi

# --- Secure compiler flags: FORTIFY_SOURCE ---
if echo "$CFLAGS" | grep -q -- "-D_FORTIFY_SOURCE=2"; then
  echo "PASS: -D_FORTIFY_SOURCE=2 in CFLAGS"
else
  echo "FAIL: -D_FORTIFY_SOURCE=2 missing from CFLAGS: $CFLAGS"
  FAILED=1
fi

# --- Secure linker flags: full RELRO ---
LDFLAGS=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS'))" 2>/dev/null)
if echo "$LDFLAGS" | grep -q -- "-Wl,-z,relro,-z,now"; then
  echo "PASS: full RELRO in LDFLAGS"
else
  echo "FAIL: -Wl,-z,relro,-z,now missing from LDFLAGS: $LDFLAGS"
  FAILED=1
fi

# --- CUDA version in path ---
if [ -d "/usr/local/cuda" ]; then
  echo "PASS: /usr/local/cuda exists"
else
  echo "FAIL: /usr/local/cuda not found"
  FAILED=1
fi

exit $FAILED
