#!/usr/bin/env bash
set -e

# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Detect GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=0
fi
echo "Detected ${NUM_GPUS} GPU(s)"

# Worker mode: join an existing Ray head instead of starting our own.
# Set RAY_ROLE=worker and RAY_HEAD_ADDRESS=<head-host>:6379 to enable.
if [ "${RAY_ROLE:-head}" = "worker" ]; then
    : "${RAY_HEAD_ADDRESS:?RAY_HEAD_ADDRESS required for worker mode (e.g. head-host:6379)}"
    # Serve replicas scheduled on this worker run user deployment code, so mirror
    # the head's Python setup: model dir on PYTHONPATH + runtime requirements.
    # The deployment code must be present at /opt/ml/model on the worker (mount a
    # shared volume, bake into the image, or use a Serve runtime_env working_dir).
    export PYTHONPATH="/opt/ml/model:${PYTHONPATH}"
    if [ -f "/opt/ml/model/code/requirements.txt" ]; then
        echo "Installing packages from /opt/ml/model/code/requirements.txt..."
        pip install -r /opt/ml/model/code/requirements.txt
    fi
    echo "Joining Ray head at ${RAY_HEAD_ADDRESS}"
    exec ray start --address="${RAY_HEAD_ADDRESS}" --num-gpus="${NUM_GPUS}" --block
fi

# Add model directory to Python path
export PYTHONPATH="/opt/ml/model:${PYTHONPATH}"

# Install runtime requirements if present
if [ -f "/opt/ml/model/code/requirements.txt" ]; then
    echo "Installing packages from /opt/ml/model/code/requirements.txt..."
    pip install -r /opt/ml/model/code/requirements.txt
fi

# Default to 127.0.0.1 to avoid exposing the serving endpoint to the network.
# Override with: docker run -e RAY_SERVE_HTTP_HOST=0.0.0.0 -e RAY_SERVE_HTTP_PORT=...
export RAY_SERVE_HTTP_HOST="${RAY_SERVE_HTTP_HOST:-127.0.0.1}"
SERVE_PORT="${RAY_SERVE_HTTP_PORT:-8000}"

# Start Ray cluster (env var must be set before this so Serve binds correctly)
ray start --head --port=6379 --include-dashboard=False --num-gpus=${NUM_GPUS}
sleep 5

# Explicitly start Serve on the correct host/port before deploying any app
serve start --http-host="${RAY_SERVE_HTTP_HOST}" --http-port="${SERVE_PORT}"
sleep 5

# Resolve serve target (config.yaml path or module:app import string)
# Falls back to /opt/ml/model/config.yaml if no CLI arg provided
if [ $# -gt 0 ]; then
    SERVE_TARGET="$1"
elif [ -f "/opt/ml/model/config.yaml" ]; then
    SERVE_TARGET="/opt/ml/model/config.yaml"
else
    echo "ERROR: No Ray Serve application found. Provide one of:"
    echo "  - docker run <image> /path/to/config.yaml"
    echo "  - docker run <image> module:app"
    echo "  - Place config at /opt/ml/model/config.yaml"
    exit 1
fi

echo "Starting Ray Serve: ${SERVE_TARGET}"
exec serve run "${SERVE_TARGET}"
