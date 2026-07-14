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

# If the customer mounted a model tarball, point HF_HOME at it so any pre-baked
# Whisper / wav2vec2 caches inside the tarball are picked up. If /opt/ml/model
# is empty (SageMaker still mounts an empty dir) fall back to the image's cache.
if [ -d /opt/ml/model ] && [ -n "$(ls -A /opt/ml/model 2>/dev/null || true)" ]; then
  export HF_HOME=/opt/ml/model
  echo "INFO: /opt/ml/model is populated; using it as HF_HOME"
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
