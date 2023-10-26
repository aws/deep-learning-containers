# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import pytest
from sagemaker.instance_group import InstanceGroup
from sagemaker.pytorch import PyTorch

from .... import invoke_pytorch_helper_function
from . import _test_mnist_distributed


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
def test_mnist_distributed_cpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend
):
    instance_type = instance_type or "ml.c4.xlarge"
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "dist_backend": dist_cpu_backend,
    }

    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
def test_mnist_distributed_gpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend
):
    instance_type = instance_type or "ml.g4dn.12xlarge"
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "dist_backend": dist_gpu_backend,
    }

    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
def test_hc_mnist_distributed_cpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend
):
    instance_type = instance_type or "ml.c4.xlarge"
    training_group = InstanceGroup("train_group", instance_type, 2)
    function_args = {
        "framework_version": framework_version,
        "instance_groups": [training_group],
        "dist_backend": dist_cpu_backend,
    }

    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.skip_pt20_cuda121_tests
def test_hc_mnist_distributed_gpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend
):
    instance_type = instance_type or "ml.g4dn.12xlarge"
    training_group = InstanceGroup("train_group", instance_type, 2)
    function_args = {
        "framework_version": framework_version,
        "instance_groups": [training_group],
        "dist_backend": dist_gpu_backend,
    }

    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_mnist_distributed, function_args
    )
