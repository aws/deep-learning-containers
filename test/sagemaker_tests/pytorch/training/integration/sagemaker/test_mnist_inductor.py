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
from sagemaker import utils
from sagemaker.instance_group import InstanceGroup
from sagemaker.pytorch import PyTorch

from ...integration import training_dir, mnist_script, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from .... import invoke_pytorch_helper_function

@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
@pytest.mark.skip_inductor_test
def test_mnist_distributed_cpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'dist_backend': dist_cpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_distributed, function_args)


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
def test_mnist_distributed_gpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'dist_backend': dist_gpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_distributed, function_args)

@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
@pytest.mark.skip_inductor_test
def test_mnist_with_native_launcher_distributed_cpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'native_launcher': True,
            'dist_backend': dist_cpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_with_native_launcher_distributed, function_args)


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
def test_mnist_with_native_launcher_distributed_gpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'native_launcher': True,
            'dist_backend': dist_gpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_mnist_with_native_launcher_distributed, function_args)


def _test_mnist_with_native_launcher_distributed(ecr_image, sagemaker_session, framework_version, instance_type, dist_backend, native_launcher):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=mnist_script,
            role='SageMakerRole',
            instance_count=2,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={'backend': dist_backend, 'epochs': 1, 'sagemaker_toolkit_native_launcher_enabled': native_launcher, 'inductor': 1},
        )
        training_input = pytorch.sagemaker_session.upload_data(path=training_dir, key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input}, job_name=utils.unique_name_from_base('test-pt-mnist-distributed'))


def _test_mnist_distributed(ecr_image, sagemaker_session, framework_version, instance_type, dist_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=mnist_script,
            role='SageMakerRole',
            instance_count=2,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={'backend': dist_backend, 'epochs': 1, 'inductor': 1},
        )
        training_input = pytorch.sagemaker_session.upload_data(path=training_dir, key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input}, job_name=utils.unique_name_from_base('test-pt-mnist-distributed'))


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_gpu
@pytest.mark.skip_inductor_test
def test_hc_mnist_distributed_cpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    training_group = InstanceGroup("train_group", instance_type, 2)
    function_args = {
            'framework_version': framework_version,
            'instance_groups': [training_group],
            'dist_backend': dist_cpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_hc_mnist_distributed, function_args)


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.integration("smexperiments")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
def test_hc_mnist_distributed_gpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    training_group = InstanceGroup("train_group", instance_type, 2)
    function_args = {
            'framework_version': framework_version,
            'instance_groups': [training_group],
            'dist_backend': dist_gpu_backend
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_hc_mnist_distributed, function_args)


def _test_hc_mnist_distributed(ecr_image, sagemaker_session, framework_version, instance_groups, dist_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=mnist_script,
            role='SageMakerRole',
            instance_groups=instance_groups,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={'backend': dist_backend, 'epochs': 1, 'inductor': 1},
        )
        training_input = pytorch.sagemaker_session.upload_data(path=training_dir, key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input}, job_name=utils.unique_name_from_base('test-pt-hc-mnist-distributed'))
