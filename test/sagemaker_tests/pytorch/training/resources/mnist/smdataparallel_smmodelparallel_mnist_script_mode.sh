#!/usr/bin/env bash
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -ex

smddpsinglenode -x RDMAV_FORK_SAFE=1 python smdataparallel_mnist.py

bash smmodelparallel_mnist_script_mode.sh
