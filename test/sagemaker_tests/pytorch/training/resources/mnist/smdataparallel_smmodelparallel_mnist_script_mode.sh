#!/usr/bin/env bash
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -ex

if [[ $(python -c "import torch; from packaging.version import Version; is_less_than_pt10 = Version(torch.__version__) < Version('1.10'); print(is_less_than_pt10)") == 'True' ]]
then
    smddpsinglenode python smdataparallel_mnist.py
else
    smddp_singlenode_dev python smdataparallel_mnist.py
fi

bash smmodelparallel_mnist_script_mode.sh
