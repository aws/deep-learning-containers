#!/usr/bin/env bash
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -ex

smddp_singlenode_dev python smdataparallel_mnist.py

bash smmodelparallel_mnist_script_mode.sh
