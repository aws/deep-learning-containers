#!/usr/bin/env bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PreRequisities
# Install mxnet >=1.3b and gluoncv on each machine
# Ensure data is kept at ~/data for each machine or change the data paths below
# Example command to start the training job
# Specify hosts in the file `hosts`

set -ex
# p3 instances have larger GPU memory, so a higher batch size can be used
# GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
# returns per GPU memory in MB
# if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=1280; else BATCH_SIZE=256; fi

# Training

python mxnet_imagenet_resnet50.py \
--rec-train $SM_CHANNEL_TRAIN/train-480px-q95.rec \
--rec-train-idx $SM_CHANNEL_TRAINIDX/train-480px-q95.idx \
--rec-val $SM_CHANNEL_VALIDATE/val-480px-q95.rec  \
--rec-val-idx $SM_CHANNEL_VALIDX/val-480px-q95.idx \
--dtype "float16" --data-nthreads "40"  \
--lr "0.3" --num-layers "50" --lr-step-epochs "20,30" \
--num-epochs "50" --batch-size 1280
