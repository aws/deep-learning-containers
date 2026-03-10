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

# Add model directory to Python path
export PYTHONPATH="/opt/ml/model:${PYTHONPATH}"

# Install runtime requirements if present
if [ -f "/opt/ml/model/code/requirements.txt" ]; then
    echo "Installing packages from /opt/ml/model/code/requirements.txt..."
    pip install -r /opt/ml/model/code/requirements.txt
fi

# Default to 0.0.0.0 so Serve is reachable outside the container.
# Can be overridden by: docker run -e RAY_SERVE_HTTP_HOST=... -e RAY_SERVE_HTTP_PORT=...
export RAY_SERVE_HTTP_HOST="${RAY_SERVE_HTTP_HOST:-0.0.0.0}"
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
