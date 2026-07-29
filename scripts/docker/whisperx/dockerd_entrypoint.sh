#!/usr/bin/env bash
# EC2 PID 1. No tini: exec uvicorn directly so signals reach it.
set -euo pipefail

# DLC telemetry: fire-and-forget IMDS ping at container start. Runs here (not
# only via bashrc) because the entrypoint exec's uvicorn directly, with no
# interactive/login shell to trigger the bashrc hook. Errors suppressed so
# telemetry never blocks or fails startup.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Activate CUDA forward-compat if the host driver is older than the baked CUDA
# libs need. No-op on new-driver and CPU hosts. Must run before uvicorn imports
# torch. We source (not `bash`) start_cuda_compat.sh so its
# `export LD_LIBRARY_PATH` lands in this shell and is inherited by the exec'd
# uvicorn. That script references $LD_LIBRARY_PATH unguarded, so pre-seed it to
# empty first — otherwise `set -u` aborts the entrypoint when it is unset (which
# it is on this image; nothing sets it before here).
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
# shellcheck source=/dev/null
. /opt/whisperx/start_cuda_compat.sh

# Any extra flags the operator passes on `docker run` land in $@; forward them
# to uvicorn (e.g. --workers 2, --log-level debug).
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir /opt/whisperx \
  "$@"
