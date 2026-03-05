#!/bin/bash
set -e

# Detect and initialize GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=0
fi

echo "Detected ${NUM_GPUS} GPU(s)"

# Handle SageMaker 'serve' command
# Default to 'serve' if no arguments provided
if [ $# -eq 0 ] || [ "$1" = "serve" ]; then
    echo "Starting Ray Serve for SageMaker..."

    # Add model directory to Python path before ray start
    export PYTHONPATH="/opt/ml/model:${PYTHONPATH}"

    # Install runtime requirements before Ray starts
    if [ -f "/opt/ml/model/code/requirements.txt" ]; then
        echo "Installing packages from /opt/ml/model/code/requirements.txt..."
        python -c "from sagemaker_serve import install_requirements; install_requirements()"
    fi

    # Start Ray cluster with detected GPU count
    ray start --head --port=6379 --include-dashboard=False --num-gpus=${NUM_GPUS}
    sleep 5

    # Start Ray Serve controller
    serve start --http-host=0.0.0.0 --http-port=8000
    sleep 5

    # Deploy Ray Serve application
    # Priority: config.yaml > SM_RAYSERVE_APP environment variable
    # Using serve run (serve deploy requires dashboard API)
    if [ -f "/opt/ml/model/config.yaml" ]; then
        echo "Running Ray Serve from default config: /opt/ml/model/config.yaml"
        serve run /opt/ml/model/config.yaml &
        RAYSERVE_PID=$!
    elif [ -n "${SM_RAYSERVE_APP}" ]; then
        echo "Running Ray Serve application: ${SM_RAYSERVE_APP}"
        serve run ${SM_RAYSERVE_APP} &
        RAYSERVE_PID=$!
    else
        echo "ERROR: No Ray Serve application found. Provide one of:"
        echo "  - /opt/ml/model/config.yaml file (default, recommended)"
        echo "  - SM_RAYSERVE_APP environment variable (format: module:app)"
        exit 1
    fi

    # Quick crash detection
    sleep 5
    if ! kill -0 $RAYSERVE_PID 2>/dev/null; then
        echo "ERROR: serve run exited immediately"
        exit 1
    fi

    # Wait for Ray Serve to be ready
    echo "Waiting for Ray Serve to start on port 8000..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/-/healthz > /dev/null 2>&1; then
            echo "Ray Serve is ready!"
            break
        fi
        echo "Waiting for Ray Serve... ($i/30)"
        sleep 2
    done

    # Start SageMaker adapter in foreground (port 8080)
    python /app/sagemaker_serve.py
else
    # If not 'serve', execute the command passed
    exec "$@"
fi
