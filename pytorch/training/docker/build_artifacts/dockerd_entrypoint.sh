#!/usr/bin/env bash

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

eval "$@"