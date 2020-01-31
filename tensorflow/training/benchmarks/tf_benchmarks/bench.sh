#!/usr/bin/env bash

execute_tensorflow_training.py train \
--framework-version  1.12 \
--device gpu \
\
--instance-types ml.p3.16xlarge \
\
--instance-counts 1 \
--instance-counts 2 \
--instance-counts 4 \
\
--py-versions py3 \
\
--subnets subnet-125fb674  \
\
--security-groups sg-ce5dd1b4  \
\
--batch-sizes 64 \
\
-- --num_batches=1000 --model vgg16 \
    --variable_update horovod --horovod_device gpu --use_fp16 --summary_verbosity 1 --save_summaries_steps 10