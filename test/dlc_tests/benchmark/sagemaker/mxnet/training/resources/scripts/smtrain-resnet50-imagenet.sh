#!/usr/bin/env bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PreRequisities
# Install mxnet >=1.3b and gluoncv on each machine
# Ensure you have Horovod, OpenMPI installed on each machine
# Specify hosts in the file `hosts`

set -ex

# Training
# num-epochs is configurable.
# Ideally user can specify num-epochs of their choice or rely on default value=90
# CodeBuild timeout is currently 90mins.
# When num-epochs = 40 and warmup-epochs = 10, training completes within 90min timeout
# Hence, for purpose of the PR num-epochs has been set to 40

python mxnet_imagenet_resnet50.py --num-epochs "40"
