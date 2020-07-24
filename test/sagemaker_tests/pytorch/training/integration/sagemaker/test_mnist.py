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
from sagemaker.pytorch import PyTorch

from ...integration import training_dir, mnist_script, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout


@pytest.mark.skip_gpu
@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode("multinode")
@pytest.mark.integration("smexperiments")
def test_mnist_distributed_cpu(sagemaker_session, ecr_image, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.skip_cpu
@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode("multinode")
@pytest.mark.integration("smexperiments")
def test_mnist_distributed_gpu(sagemaker_session, ecr_image, instance_type, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_gpu_backend)


def _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, dist_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point=mnist_script,
                          role='SageMakerRole',
                          train_instance_count=2,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session,
                          image_name=ecr_image,
                          hyperparameters={'backend': dist_backend, 'epochs': 1})
        training_input = pytorch.sagemaker_session.upload_data(path=training_dir,
                                                               key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input})
