
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

python -W ignore deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py \
--data_dir $SM_CHANNEL_TRAIN --num_epochs 90 -b $BATCH_SIZE \
--lr_decay_mode poly --warmup_epochs 10 --clear_log
