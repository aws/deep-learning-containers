#!/bin/bash

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "HF_MODEL_ID must be set"
    exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
    export REVISION="${HF_MODEL_REVISION}"
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

# NOTE: When the installed NVIDIA kernel driver is older than the version required
# by the CUDA compat package (as indicated by the libcuda.so.1 symlink), we need
# to include the compat directory in `LD_LIBRARY_PATH` to bridge the mismatch.
if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d'.' -f 3-)
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    if [ "$NVIDIA_DRIVER_VERSION" != "$CUDA_COMPAT_MAX_DRIVER_VERSION" ] &&
        [ "$NVIDIA_DRIVER_VERSION" = "$(printf '%s\n' "$NVIDIA_DRIVER_VERSION" "$CUDA_COMPAT_MAX_DRIVER_VERSION" | sort -V | head -n1)" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"
    fi
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 --port 8080 --json-output
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec text-embeddings-router-80 --port 8080 --json-output
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 --port 8080 --json-output
elif [ ${compute_cap} -eq 100 ]; then
    exec text-embeddings-router-100 --port 8080 --json-output
elif [ ${compute_cap} -eq 120 ]; then
    exec text-embeddings-router-120 --port 8080 --json-output
else
    echo "cuda compute cap ${compute_cap} is not supported"
    exit 1
fi
