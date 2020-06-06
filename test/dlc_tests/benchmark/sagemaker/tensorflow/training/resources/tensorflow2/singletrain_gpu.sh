#!/usr/bin/env bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Ensure you have Horovod, OpenMPI, and Tensorflow installed on each machine
# Specify hosts in the file `hosts`

set -ex

# p3 instances have larger GPU memory, so a higher batch size can be used
GPU_MEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}'`
if [ $GPU_MEM -gt 15000 ] ; then BATCH_SIZE=256; else BATCH_SIZE=128; fi

# Training

python -W ignore deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py \
--data_dir $SM_CHANNEL_TRAIN --synthetic --batch_size 128 --num_batches 100 --clear_log 2
