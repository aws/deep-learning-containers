#!/bin/bash

TF_SERVING_PACKAGE=$(pip list | grep tensorflow-serving | cut -d ' ' -f 1)

if [[ ${TF_SERVING_PACKAGE} == *"gpu"* ]]; then
  bash /usr/local/bin/start_cuda_compat.sh
fi

eval "$@"
