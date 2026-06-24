#!/usr/bin/env bash
#
# SOURCE this from an entrypoint (it must export LD_LIBRARY_PATH into the parent
# shell). It activates the CUDA forward-compatibility libraries when the host
# NVIDIA driver is older than the toolkit baked into the image; if the host
# driver is new enough, the compat layer is skipped and the host driver is used.
#
# Source-safe: defines a function and uses `return`, with no global `set -e`,
# so it never terminates the entrypoint that sources it.

_hf_start_cuda_compat() {
  local compat_dir="/usr/local/cuda/compat"

  if [[ ! -d "${compat_dir}" ]]; then
    echo "[cuda-compat] no compat libraries present, using host driver"
    return 0
  fi

  local compat_ver host_ver
  compat_ver="$(cat "${compat_dir}/version" 2>/dev/null || echo "0")"
  host_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "0")"
  echo "[cuda-compat] host driver=${host_ver:-unknown} compat driver=${compat_ver}"

  # verlte A B  -> true when A <= B
  if [[ -n "${host_ver}" ]] && [[ "${compat_ver}" == "$(printf '%s\n%s' "${compat_ver}" "${host_ver}" | sort -V | head -n1)" ]]; then
    echo "[cuda-compat] host driver is new enough, skipping compat layer"
  else
    echo "[cuda-compat] enabling forward-compat libraries from ${compat_dir}"
    export LD_LIBRARY_PATH="${compat_dir}:${LD_LIBRARY_PATH:-}"
  fi
}

_hf_start_cuda_compat
