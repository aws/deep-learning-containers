#!/bin/bash
# CUDA forward-compat activation. Local copy of the canonical DLC PyTorch script
# (scripts/docker/pytorch/start_cuda_compat.sh); kept whisperx-local so this
# image does not couple to the training-fleet script. Keep the logic in sync.
#
# Prepends the bundled cuda-compat userspace libcuda (/usr/local/cuda/compat) to
# LD_LIBRARY_PATH only when the host NVIDIA driver is older than the compat build
# needs; newer-driver hosts and CPU hosts (no compat package) are left untouched.
#
# CALLER CONTRACT: the whisperx entrypoints `source` this under `set -u` so the
# exported LD_LIBRARY_PATH reaches the exec'd uvicorn. The `$LD_LIBRARY_PATH`
# reference below is unguarded, so callers MUST pre-seed it (they run
# `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"` first); otherwise sourcing
# aborts on an unbound variable when compat activates.

verlte() {
  [ "$1" = "$2" ] && return 1 || [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
  CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d'.' -f 3-)
  echo "CUDA compat package should be installed for NVIDIA driver smaller than ${CUDA_COMPAT_MAX_DRIVER_VERSION}"
  NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
  if [ -z "$NVIDIA_DRIVER_VERSION" ]; then
    NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 2>/dev/null || true)
  fi
  echo "Current installed NVIDIA driver version is ${NVIDIA_DRIVER_VERSION}"
  if verlte $NVIDIA_DRIVER_VERSION $CUDA_COMPAT_MAX_DRIVER_VERSION; then
    echo "Adding CUDA compat to LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH
  else
    echo "Skipping CUDA compat setup as newer NVIDIA driver is installed"
  fi
else
  echo "Skipping CUDA compat setup as package not found"
fi
