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

from . import skip_if_not_v3_compatible, _test_mnist_distributed_v3


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
@pytest.mark.team("conda")
def test_mnist_distributed_cpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.c5.xlarge"
    _test_mnist_distributed_v3(
        ecr_image,
        sagemaker_regions,
        framework_version=framework_version,
        dist_backend=dist_cpu_backend,
        instance_type=instance_type,
    )


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.team("conda")
def test_mnist_distributed_gpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.g4dn.12xlarge"
    _test_mnist_distributed_v3(
        ecr_image,
        sagemaker_regions,
        framework_version=framework_version,
        dist_backend=dist_gpu_backend,
        instance_type=instance_type,
    )


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
@pytest.mark.team("conda")
def test_hc_mnist_distributed_cpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.c5.xlarge"
    _test_mnist_distributed_v3(
        ecr_image,
        sagemaker_regions,
        framework_version=framework_version,
        dist_backend=dist_cpu_backend,
        instance_type=instance_type,
        instance_count=2,
    )


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.team("conda")
def test_hc_mnist_distributed_gpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.g4dn.12xlarge"
    _test_mnist_distributed_v3(
        ecr_image,
        sagemaker_regions,
        framework_version=framework_version,
        dist_backend=dist_gpu_backend,
        instance_type=instance_type,
        instance_count=2,
    )
