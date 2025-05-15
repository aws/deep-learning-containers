#!/usr/bin/env bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh 2>/dev/null || true

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

eval "$@"