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
--dtype "float16" --data-nthreads "40"  \
--lr "0.3" --warmup-epochs 0 \
--num-epochs "1" --batch-size 256 --mode "gluon"
