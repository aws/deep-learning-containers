#!/usr/bin/env bash
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

./execute_tensorflow_training.py train \
--framework-version  1.12 \
--device gpu \
\
--instance-types ml.p3.16xlarge \
\
--instance-counts 1 \
--instance-counts 2 \
--instance-counts 4 \
--instance-counts 8 \
--instance-counts 16 \
\
--py-versions py3 \
\
--subnets # add subnet id here  \
\
--security-groups # add security-group id here
