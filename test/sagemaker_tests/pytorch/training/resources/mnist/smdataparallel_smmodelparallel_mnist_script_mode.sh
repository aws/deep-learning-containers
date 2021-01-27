#!/usr/bin/env bash
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -ex

mpirun -mca btl_vader_single_copy_mechanism none --allow-run-as-root -np 8 python smmodelparallel_pt_mnist.py --num-microbatches 4 --pipeline interleaved --ddp 1 --assert-losses 1

smddpsinglenode python smdataparallel_mnist.py
