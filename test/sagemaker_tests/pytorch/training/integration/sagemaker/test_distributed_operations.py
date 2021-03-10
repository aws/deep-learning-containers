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
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from ...integration import (data_dir, dist_operations_path, fastai_path, mnist_script,
                              DEFAULT_TIMEOUT, mnist_path)
from ...integration.sagemaker.timeout import timeout

MULTI_GPU_INSTANCE = 'ml.p3.8xlarge'


@pytest.mark.processor("cpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
@pytest.mark.deploy_test
@pytest.mark.skip_test_in_region
def test_dist_operations_cpu(sagemaker_session, framework_version, ecr_image, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    _test_dist_operations(sagemaker_session, framework_version, ecr_image, instance_type, dist_cpu_backend)


@pytest.mark.processor("gpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
@pytest.mark.deploy_test
def test_dist_operations_gpu(sagemaker_session, framework_version, instance_type, ecr_image, dist_gpu_backend):
    """
    Test is run as multinode
    """
    instance_type = instance_type or 'ml.p2.xlarge'
    _test_dist_operations(sagemaker_session, framework_version, ecr_image, instance_type, dist_gpu_backend)


@pytest.mark.processor("gpu")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
def test_dist_operations_multi_gpu(sagemaker_session, framework_version, ecr_image, dist_gpu_backend):
    """
    Test is run as single node, but multi-gpu
    """
    _test_dist_operations(sagemaker_session, framework_version, ecr_image, MULTI_GPU_INSTANCE, dist_gpu_backend, 1)


@pytest.mark.processor("gpu")
@pytest.mark.integration("fastai")
@pytest.mark.model("cifar")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_dist_operations_fastai_gpu(sagemaker_session, framework_version, ecr_image):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point='train_cifar.py',
            source_dir=os.path.join(fastai_path, 'cifar'),
            role='SageMakerRole',
            instance_count=1,
            instance_type=MULTI_GPU_INSTANCE,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
        )
        pytorch.sagemaker_session.default_bucket()
        training_input = pytorch.sagemaker_session.upload_data(
            path=os.path.join(fastai_path, 'cifar_tiny', 'training'), key_prefix='pytorch/distributed_operations'
        )
        pytorch.fit({'training': training_input})

    model_s3_url = pytorch.create_model().model_data
    _assert_s3_file_exists(sagemaker_session.boto_region_name, model_s3_url)


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_mnist_gpu(sagemaker_session, framework_version, ecr_image, dist_gpu_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=mnist_script,
            role='SageMakerRole',
            image_uri=ecr_image,
            instance_count=2,
            framework_version=framework_version,
            instance_type=MULTI_GPU_INSTANCE,
            sagemaker_session=sagemaker_session,
            hyperparameters={'backend': dist_gpu_backend},
        )

        training_input = sagemaker_session.upload_data(
            path=os.path.join(data_dir, 'training'), key_prefix='pytorch/mnist'
        )
        pytorch.fit({'training': training_input})


@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smmodelparallel_mnist_multigpu(ecr_image, instance_type, py_version, sagemaker_session, tmpdir):
    """
    Tests pt mnist command via script mode
    """
    instance_type = "ml.p3.16xlarge"
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if not(Version(image_framework_version) in SpecifierSet(">=1.6,<1.8")) or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on PyTorch 1.6 and PyTorch 1.7")

    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point='smmodelparallel_pt_mnist.sh',
                          role='SageMakerRole',
                          image_uri=ecr_image,
                          source_dir=mnist_path,
                          instance_count=1,
                          instance_type=instance_type,
                          sagemaker_session=sagemaker_session)

        pytorch.fit()


@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smmodelparallel_mnist_multigpu_multinode(ecr_image, instance_type, py_version, sagemaker_session, tmpdir):
    """
    Tests pt mnist command via script mode
    """
    instance_type = "ml.p3.16xlarge"
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if not(Version(image_framework_version) in SpecifierSet(">=1.6,<1.8")) or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on PyTorch 1.6 and PyTorch 1.7")

    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point='smmodelparallel_pt_mnist_multinode.sh',
                          role='SageMakerRole',
                          image_uri=ecr_image,
                          source_dir=mnist_path,
                          instance_count=2,
                          instance_type=instance_type,
                          sagemaker_session=sagemaker_session)

        pytorch.fit()


@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip(reason="Skipping test because it is flaky on mainline pipeline.")
def test_smdataparallel_mnist_script_mode_multigpu(ecr_image, instance_type, py_version, sagemaker_session, tmpdir):
    """
    Tests SM Distributed DataParallel single-node via script mode
    """
    instance_type = "ml.p3.16xlarge"
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point='smdataparallel_mnist_script_mode.sh',
                          role='SageMakerRole',
                          image_uri=ecr_image,
                          source_dir=mnist_path,
                          instance_count=1,
                          instance_type=instance_type,
                          sagemaker_session=sagemaker_session)

        pytorch.fit()


@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('instance_types', ["ml.p3.16xlarge", "ml.p3dn.24xlarge"])
def test_smdataparallel_mnist(instance_types, ecr_image, py_version, sagemaker_session, tmpdir):
    """
    Tests smddprun command via Estimator API distribution parameter
    """
    distribution = {"smdistributed":{"dataparallel":{"enabled":True}}}
    estimator = PyTorch(entry_point='smdataparallel_mnist.py',
                        role='SageMakerRole',
                        image_uri=ecr_image,
                        source_dir=mnist_path,
                        instance_count=2,
                        instance_type=instance_types,
                        sagemaker_session=sagemaker_session,
                        distribution=distribution)

    estimator.fit()


@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.integration("smdataparallel_smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.parametrize('instance_types', ["ml.p3.16xlarge"])
def test_smmodelparallel_smdataparallel_mnist(instance_types, ecr_image, py_version, sagemaker_session, tmpdir):
    """
    Tests SM Distributed DataParallel and ModelParallel single-node via script mode
    This test has been added for SM DataParallelism and ModelParallelism tests for re:invent.
    TODO: Consider reworking these tests after re:Invent releases are done
    """
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if not (Version(image_framework_version) in SpecifierSet(">=1.6,<1.8")) or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on PyTorch 1.6 and PyTorch 1.7")
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point='smdataparallel_smmodelparallel_mnist_script_mode.sh',
                          role='SageMakerRole',
                          image_uri=ecr_image,
                          source_dir=mnist_path,
                          instance_count=1,
                          instance_type=instance_types,
                          sagemaker_session=sagemaker_session)

        pytorch = _disable_sm_profiler(sagemaker_session.boto_region_name, pytorch)

        pytorch.fit()


def _test_dist_operations(
        sagemaker_session, framework_version, ecr_image, instance_type, dist_backend, instance_count=3
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=dist_operations_path,
            role='SageMakerRole',
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={'backend': dist_backend},
        )

        pytorch = _disable_sm_profiler(sagemaker_session.boto_region_name, pytorch)
        
        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=dist_operations_path, key_prefix='pytorch/distributed_operations'
        )
        pytorch.fit({'required_argument': fake_input})


def _assert_s3_file_exists(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3', region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()

def _disable_sm_profiler(region, estimator):
    """Disable SMProfiler feature for China regions
    """

    if region in ('cn-north-1', 'cn-northwest-1'):
        estimator.disable_profiler = True
    return estimator
