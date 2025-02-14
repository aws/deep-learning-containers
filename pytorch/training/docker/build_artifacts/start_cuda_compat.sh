#!/usr/bin/env bash

verlte() {
  [ "$1" = "$2" ] && return 1 || [ "$1" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

if [ -f ${CUDA_HOME}/compat/libcuda.so.1 ]; then
  CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink ${CUDA_HOME}/compat/libcuda.so.1 | cut -d'.' -f 3-)
  echo "CUDA compat package requires Nvidia driver ⩽${CUDA_COMPAT_MAX_DRIVER_VERSION}"
  NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 2>/dev/null || true)
  echo "Current installed Nvidia driver version is ${NVIDIA_DRIVER_VERSION}"
  if verlte $NVIDIA_DRIVER_VERSION $CUDA_COMPAT_MAX_DRIVER_VERSION; then
    echo "Setup CUDA compatibility libs path to LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH
  else
    echo "Skip CUDA compat libs setup as newer Nvidia driver is installed"
  fi
else
  echo "Skip CUDA compat libs setup as package not found"
fi
