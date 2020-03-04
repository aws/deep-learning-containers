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

import os

import boto3
import pytest
from sagemaker.pytorch import PyTorch
from six.moves.urllib.parse import urlparse

from ...integration import (data_dir, dist_operations_path, fastai_path, mnist_script,
                              DEFAULT_TIMEOUT)
from ...integration.sagemaker.timeout import timeout

MULTI_GPU_INSTANCE = 'ml.p3.8xlarge'


@pytest.mark.skip_gpu
@pytest.mark.deploy_test
@pytest.mark.skip_test_in_region
def test_dist_operations_cpu(sagemaker_session, ecr_image, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.skip_cpu
@pytest.mark.deploy_test
def test_dist_operations_gpu(sagemaker_session, instance_type, ecr_image, dist_gpu_backend):
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_gpu_backend)


@pytest.mark.skip_cpu
def test_dist_operations_multi_gpu(sagemaker_session, ecr_image, dist_gpu_backend):
    _test_dist_operations(sagemaker_session, ecr_image, MULTI_GPU_INSTANCE, dist_gpu_backend, 1)


@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_dist_operations_fastai_gpu(sagemaker_session, ecr_image):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point='train_cifar.py',
                          source_dir=os.path.join(fastai_path, 'cifar'),
                          role='SageMakerRole',
                          train_instance_count=1,
                          train_instance_type=MULTI_GPU_INSTANCE,
                          sagemaker_session=sagemaker_session,
                          image_name=ecr_image)
        pytorch.sagemaker_session.default_bucket()
        training_input = pytorch.sagemaker_session.upload_data(
            path=os.path.join(fastai_path, 'cifar_tiny', 'training'),
            key_prefix='pytorch/distributed_operations'
        )
        pytorch.fit({'training': training_input})

    model_s3_url = pytorch.create_model().model_data
    _assert_s3_file_exists(sagemaker_session.boto_region_name, model_s3_url)


@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_mnist_gpu(sagemaker_session, ecr_image, dist_gpu_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point=mnist_script,
                          role='SageMakerRole',
                          train_instance_count=2,
                          image_name=ecr_image,
                          train_instance_type=MULTI_GPU_INSTANCE,
                          sagemaker_session=sagemaker_session,
                          hyperparameters={'backend': dist_gpu_backend})

        training_input = sagemaker_session.upload_data(path=os.path.join(data_dir, 'training'),
                                                       key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input})


def _test_dist_operations(sagemaker_session, ecr_image, instance_type, dist_backend, train_instance_count=3):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point=dist_operations_path,
                          role='SageMakerRole',
                          train_instance_count=train_instance_count,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session,
                          image_name=ecr_image,
                          hyperparameters={'backend': dist_backend})
        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(path=dist_operations_path,
                                                           key_prefix='pytorch/distributed_operations')
        pytorch.fit({'required_argument': fake_input})


def _assert_s3_file_exists(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3', region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()
