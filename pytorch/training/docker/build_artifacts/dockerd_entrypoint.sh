#!/usr/bin/env bash
bash /usr/local/bin/bash_telemetry.sh

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

eval "$@"