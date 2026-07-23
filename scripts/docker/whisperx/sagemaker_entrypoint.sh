#!/usr/bin/env bash
# SageMaker PID 1. SageMaker invokes the container with `serve` as the first
# argument; anything else means the user is `docker run …`-ing this variant
# outside SageMaker, in which case we behave identically to the EC2 target.
set -euo pipefail

# DLC telemetry: fire-and-forget IMDS ping at container start. Runs here (not
# only via bashrc) because the entrypoint exec's uvicorn directly, with no
# interactive/login shell to trigger the bashrc hook. Errors suppressed so
# telemetry never blocks or fails startup.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Model source resolution (mirrors the vLLM SageMaker entrypoint's ladder):
#   1. WHISPERX_DEFAULT_MODEL already set    -> explicit override, respect it.
#   2. /opt/ml/model populated (SageMaker    -> serve it directly. SageMaker
#      staged the customer's ModelDataUrl        extracts ModelDataUrl here; a
#      here)                                      customer model dir is the common
#                                                 bring-your-own-model path.
#   3. neither                                -> image default (large-v2 in server.py).
#
# For (2) we point WHISPERX_DEFAULT_MODEL at the dir itself: faster-whisper's
# WhisperModel loads an existing directory path directly (its os.path.isdir
# branch), so a flat model dir works without needing an HF-cache layout. HF_HOME
# is also repointed so an HF-cache-layout tarball (or bundled aligner caches) is
# still picked up. If /opt/ml/model is empty (SageMaker mounts an empty dir when
# no ModelDataUrl is given) both are skipped and the image default is used.
if [ -d /opt/ml/model ] && [ -n "$(ls -A /opt/ml/model 2>/dev/null || true)" ]; then
  export HF_HOME=/opt/ml/model
  if [ -z "${WHISPERX_DEFAULT_MODEL:-}" ]; then
    export WHISPERX_DEFAULT_MODEL=/opt/ml/model
    echo "INFO: /opt/ml/model is populated; serving it (WHISPERX_DEFAULT_MODEL=/opt/ml/model)"
  else
    echo "INFO: /opt/ml/model populated but WHISPERX_DEFAULT_MODEL=${WHISPERX_DEFAULT_MODEL} is set; respecting override"
  fi
fi

# Activate CUDA forward-compat if the host driver is older than the baked CUDA
# libs need (e.g. SageMaker ml.g4dn ships an older driver). No-op on new-driver
# and CPU hosts. Must run before uvicorn imports torch. See cuda_compat.sh.
# shellcheck source=/dev/null
. /opt/whisperx/cuda_compat.sh

# SageMaker passes `serve` when it launches an inference endpoint. Any other
# invocation (including no args at all) is a local docker run — treat it the
# same, since the entrypoint has nothing else to do.
if [ "${1:-serve}" = "serve" ]; then
  shift || true
fi

exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --app-dir /opt/whisperx \
  "$@"
