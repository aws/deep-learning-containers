#!/usr/bin/env bash
set -euo pipefail

# Preserve an explicit network selection. Otherwise exclude loopback and the
# Docker bridge so NCCL can select the EC2 data-plane interface.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="^docker0,lo"
fi

# Source rather than execute so a compatible libcuda path remains exported.
# The shared script is not nounset-safe: with no visible NVIDIA driver it calls
# verlte with one argument and would abort before the command runs.
if [[ -f /usr/local/bin/start_cuda_compat.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/bin/start_cuda_compat.sh
  set -u
fi

exec "$@"
