#!/usr/bin/env bash
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -ex


mpirun --allow-run-as-root -np 2 python smmodelparallel_pt_mnist.py --num-microbatches 4 --pipeline simple --assert-losses 1

mpirun -mca btl_vader_single_copy_mechanism none --allow-run-as-root -np 2 python smmodelparallel_pt_mnist.py --num-microbatches 4 --pipeline interleaved --assert-losses 1

mpirun -mca btl_vader_single_copy_mechanism none --allow-run-as-root -np 8 python smmodelparallel_pt_mnist.py --num-microbatches 4 --pipeline interleaved --horovod 1 --assert-losses 1

mpirun -mca btl_vader_single_copy_mechanism none --allow-run-as-root -np 8 python smmodelparallel_pt_mnist.py --num-microbatches 4 --pipeline interleaved --horovod 1 --amp 1 --assert-losses 1

