#!/usr/bin/env bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PreRequisities
# Install mxnet >=1.3b and gluoncv on each machine

echo "I'm a sm train bash"

# Get release version on mxnet container
MXNET_VERSION=$(python -c "import mxnet; print(mxnet.__version__)")

# Clone source to get launch.py script to start training job
git clone --recursive -b ${MXNET_VERSION} https://github.com/apache/incubator-mxnet

# Ensure data is kept at ~/data for each machine or change the data paths below
# Example command to start the training job
# Specify hosts in the file `hosts`

LAST_GPU=$(( $SM_NUM_GPUS - 1 ))

GPUS=$(seq -s "," 0 $LAST_GPU)

echo "Running on GPUs" $GPUS

python incubator-mxnet/example/image-classification/train_imagenet.py \
--data-train  $SM_CHANNEL_TRAIN/train-480px-q95.rec \
--data-train-idx $SM_CHANNEL_TRAINIDX/train-480px-q95.idx \
--data-val   $SM_CHANNEL_VALIDATE/val-480px-q95.rec  \
--data-val-idx $SM_CHANNEL_VALIDX/val-480px-q95.idx \
--dtype  float16  --data-nthreads  "40"  \
--lr  "0.3" --kv-store  dist_device_sync  --network  resnet  --num-layers "50" --lr-step-epochs \
  "30,60,90" --num-epochs  "120" --batch-size  "1280" --benchmark '0' --gpus $GPUS
