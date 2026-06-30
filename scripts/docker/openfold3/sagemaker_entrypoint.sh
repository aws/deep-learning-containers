#!/bin/bash
# OpenFold3 SageMaker entrypoint: gunicorn-served Flask app (/ping + /invocations).

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

echo "Starting OpenFold3 server"

# workers=1 is mandatory: the GPU pool is per-process, so a second worker would
# also try to claim all GPUs. Threads handle concurrent requests within the pool.
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers 1 \
    --threads 8 \
    --worker-class gthread \
    --timeout 3600 \
    --graceful-timeout 60 \
    --access-logfile - \
    --error-logfile - \
    sagemaker_serve:app
