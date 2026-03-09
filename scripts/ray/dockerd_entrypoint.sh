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

# Start Ray cluster
ray start --head --port=6379 --include-dashboard=False --num-gpus=${NUM_GPUS}
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
