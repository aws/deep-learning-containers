#!/usr/bin/env bash
# EC2 PID 1. Design §3: no tini, exec uvicorn directly so signals reach it.
set -euo pipefail

# Activate CUDA forward-compat if the host driver is older than the baked CUDA
# libs need. No-op on new-driver and CPU hosts. Must run before uvicorn imports
# torch. See cuda_compat.sh.
# shellcheck source=/dev/null
. /opt/whisperx/cuda_compat.sh

# Any extra flags the operator passes on `docker run` land in $@; forward them
# to uvicorn (e.g. --workers 2, --log-level debug).
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir /opt/whisperx \
  "$@"
