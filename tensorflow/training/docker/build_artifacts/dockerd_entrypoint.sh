#!/usr/bin/env bash

CUDA_AVAILABLE=$(python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

eval "$@"
