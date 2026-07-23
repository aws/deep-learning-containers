#!/usr/bin/env bash
# CUDA forward-compatibility activation. Sourced by both entrypoints BEFORE they
# exec uvicorn.
#
# Why this exists: the image bakes torch's cu128 CUDA libs. On a GPU host whose
# NVIDIA driver is older than those libs need, this prepends the bundled
# cuda-compat userspace libcuda (/usr/local/cuda/compat) to LD_LIBRARY_PATH so
# CUDA still initialises — but ONLY when the host driver is actually older than
# the compat build, so newer-driver hosts (and CPU hosts with no driver) are
# untouched. Adapted from DLC scripts/docker/pytorch/{entrypoint,start_cuda_compat}.sh.
#
# IMPORTANT — SageMaker GPU endpoints need BOTH this AND an AMI pin:
#   cuda-compat can only bridge from a base driver that itself meets the compat
#   package's minimum. SageMaker's DEFAULT g4dn host AMI ships a driver too old
#   even for that, so the container fails to start with a zero-log
#   "CannotStartContainerError" REGARDLESS of this script. The endpoint-config
#   MUST set InferenceAmiVersion to a CUDA-12.x AMI:
#       "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1"   # driver 550
#   With that AMI the host driver is 550.163.01; this script then activates
#   compat (550 < 570.211.01) and CUDA 12.8 runs. VERIFIED on ml.g4dn.xlarge:
#   endpoint reached InService and served GPU transcription + diarization.
#   (CUDA 13.x images use al2023-ami-sagemaker-inference-gpu-4-1, driver 580.)
#   See DLC test/test_utils/constants.py for the CUDA-major -> AMI mapping.
#
# This is intentionally guarded to never abort a caller running under
# `set -euo pipefail`: every probe that can legitimately fail (no GPU, no
# /proc/driver/nvidia) is `|| true`-guarded and the whole thing is a no-op when
# the compat lib is absent or the host driver is already new enough.

_activate_cuda_forward_compat() {
  local compat_file=/usr/local/cuda/compat/libcuda.so.1
  [ -f "$compat_file" ] || return 0

  # Max driver the compat build speaks, parsed from e.g.
  # libcuda.so.1 -> libcuda.so.570.211.01  =>  570.211.01
  local compat_max
  compat_max="$(readlink "$compat_file" 2>/dev/null | cut -d'.' -f3- || true)"
  [ -n "$compat_max" ] || return 0

  # Host driver: prefer /proc (present iff a GPU + kernel module exist), then
  # nvidia-smi. On a CPU host neither exists -> empty -> skip (no compat needed).
  local host_drv=""
  host_drv="$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)"
  if [ -z "$host_drv" ]; then
    host_drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0 2>/dev/null || true)"
  fi
  [ -n "$host_drv" ] || return 0

  # Activate compat ONLY when host_drv < compat_max (strict). If the host driver
  # is >= the compat build, the host driver is already sufficient — leave it.
  local lowest
  lowest="$(printf '%s\n%s\n' "$host_drv" "$compat_max" | sort -V | head -n1 || true)"
  if [ "$host_drv" = "$lowest" ] && [ "$host_drv" != "$compat_max" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
    echo "INFO: host NVIDIA driver ${host_drv} < CUDA compat ${compat_max}; activated forward-compat libcuda (/usr/local/cuda/compat)"
  else
    echo "INFO: host NVIDIA driver ${host_drv} >= CUDA compat ${compat_max}; using host driver (no forward-compat needed)"
  fi
}

_activate_cuda_forward_compat
