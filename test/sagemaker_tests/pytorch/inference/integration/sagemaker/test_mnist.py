# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import numpy as np
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel

from test.test_utils import ML_Model
from ...integration import model_cpu_dir, mnist_cpu_script, mnist_gpu_script, model_eia_dir, mnist_eia_script
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.cpu_test
@pytest.mark.model(ML_Model.MNIST.value)
@pytest.mark.multinode("multinode")
def test_mnist_distributed_cpu(sagemaker_session, ecr_image, instance_type):
    instance_type = instance_type or 'ml.c4.xlarge'
    model_dir = os.path.join(model_cpu_dir, 'model_mnist.tar.gz')
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, model_dir, mnist_cpu_script)


@pytest.mark.gpu_test
@pytest.mark.model(ML_Model.MNIST.value)
@pytest.mark.multinode("multinode")
def test_mnist_distributed_gpu(sagemaker_session, ecr_image, instance_type):
    instance_type = instance_type or 'ml.p2.xlarge'
    model_dir = os.path.join(model_cpu_dir, 'model_mnist.tar.gz')
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, model_dir, mnist_gpu_script)


@pytest.mark.eia_test
@pytest.mark.model(ML_Model.MNIST.value)
@pytest.mark.integration("elastic_inference")
@pytest.mark.processor("eia")
def test_mnist_eia(sagemaker_session, ecr_image, instance_type, accelerator_type):
    instance_type = instance_type or 'ml.c4.xlarge'
    # Scripted model is serialized with torch.jit.save().
    # Inference test for EIA doesn't need to instantiate model definition then load state_dict
    model_dir = os.path.join(model_eia_dir, 'model_mnist.tar.gz')
    _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, model_dir, mnist_eia_script,
                            accelerator_type=accelerator_type)


def _test_mnist_distributed(sagemaker_session, ecr_image, instance_type, model_dir, mnist_script,
                            accelerator_type=None):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(model_data=model_data, role='SageMakerRole', entry_point=mnist_script,
                           image=ecr_image, sagemaker_session=sagemaker_session)

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        # Use accelerator type to differentiate EI vs. CPU and GPU. Don't use processor value
        if accelerator_type is not None:
            predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type,
                                       accelerator_type=accelerator_type, endpoint_name=endpoint_name)
        else:
            predictor = pytorch.deploy(initial_instance_count=1, instance_type=instance_type,
                                       endpoint_name=endpoint_name)

        batch_size = 100
        data = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
        output = predictor.predict(data)

        assert output.shape == (batch_size, 10)
