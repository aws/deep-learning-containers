#!/usr/bin/env bash
set -euo pipefail

# Preserve an explicit network selection. Otherwise exclude loopback and the
# Docker bridge so NCCL can select the EC2 data-plane interface.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="^docker0,lo"
fi

# Source rather than execute so a compatible libcuda path remains exported.
if [[ -f /usr/local/bin/start_cuda_compat.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/bin/start_cuda_compat.sh
fi

exec "$@"
