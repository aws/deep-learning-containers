#!/bin/bash
# entrypoint.sh — CUDA forward compatibility check, then exec user command
set -e

# CUDA forward compatibility
COMPAT_FILE=/usr/local/cuda/compat/libcuda.so.1
if [ -f "$COMPAT_FILE" ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink "$COMPAT_FILE" | cut -d'.' -f 3-)
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    if [ -z "$NVIDIA_DRIVER_VERSION" ]; then
        NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 2>/dev/null || true)
    fi
    if [ -n "$NVIDIA_DRIVER_VERSION" ] && [ -n "$CUDA_COMPAT_MAX_DRIVER_VERSION" ]; then
        if [ "$NVIDIA_DRIVER_VERSION" = "$(echo -e "$NVIDIA_DRIVER_VERSION\n$CUDA_COMPAT_MAX_DRIVER_VERSION" | sort -V | head -n1)" ] \
           && [ "$NVIDIA_DRIVER_VERSION" != "$CUDA_COMPAT_MAX_DRIVER_VERSION" ]; then
            export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}
        fi
    fi
fi

exec "$@"
