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
from sagemaker import utils
from sagemaker.pytorch import PyTorch
from sagemaker import Session
from six.moves.urllib.parse import urlparse
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import (data_dir, dist_operations_path, fastai_path, mnist_script,
                              DEFAULT_TIMEOUT, mnist_path)
from ...integration.sagemaker.timeout import timeout
from .... import invoke_pytorch_helper_function
from . import invoke_pytorch_estimator

MULTI_GPU_INSTANCE = 'ml.p3.8xlarge'
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


def validate_or_skip_smmodelparallel(ecr_image):
    if not can_run_smmodelparallel(ecr_image):
        pytest.skip("Model Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smmodelparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(
        image_cuda_version.strip("cu")) >= Version("110")


def validate_or_skip_smmodelparallel_efa(ecr_image):
    if not can_run_smmodelparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on PyTorch 1.8.1 or higher")


def can_run_smmodelparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.8.1") and Version(image_cuda_version.strip("cu")) >= Version("110")


@pytest.mark.processor("cpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
@pytest.mark.deploy_test
@pytest.mark.skip_test_in_region
def test_dist_operations_cpu(framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend):
    instance_type = instance_type or 'ml.c4.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'dist_backend': dist_cpu_backend,
        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_dist_operations, function_args)


@pytest.mark.processor("gpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
@pytest.mark.deploy_test
def test_dist_operations_gpu(framework_version, instance_type, ecr_image, sagemaker_regions, dist_gpu_backend):
    """
    Test is run as multinode
    """
    instance_type = instance_type or 'ml.p2.xlarge'
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'dist_backend': dist_gpu_backend,
        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_dist_operations, function_args)

@pytest.mark.processor("gpu")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
def test_dist_operations_multi_gpu(framework_version, ecr_image, sagemaker_regions, dist_gpu_backend):
    """
    Test is run as single node, but multi-gpu
    """
    function_args = {
            'framework_version': framework_version,
            'instance_type': MULTI_GPU_INSTANCE,
            'dist_backend': dist_gpu_backend,
            'instance_count': 1
        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_dist_operations, function_args)

@pytest.mark.processor("gpu")
@pytest.mark.integration("fastai")
@pytest.mark.model("cifar")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_dist_operations_fastai_gpu(framework_version, ecr_image, sagemaker_regions):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    if Version("1.9") <= Version(image_framework_version) < Version("1.11"):
        pytest.skip("Fast ai is not supported on PyTorch v1.9.x and v1.10.x")

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            'entry_point': 'train_cifar.py',
            'source_dir': os.path.join(fastai_path, 'cifar'),
            'role': 'SageMakerRole',
            'instance_count': 1,
            'instance_type': MULTI_GPU_INSTANCE,
            'framework_version': framework_version,
        }
        upload_s3_data_args = {
        'path': os.path.join(fastai_path, 'cifar_tiny', 'training'),
        'key_prefix': 'pytorch/distributed_operations'
        }

        job_name=utils.unique_name_from_base('test-pt-fastai')
        pytorch, sagemaker_session = invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, upload_s3_data_args=upload_s3_data_args, job_name=job_name)

    model_s3_url = pytorch.create_model().model_data
    _assert_s3_file_exists(sagemaker_session.boto_region_name, model_s3_url)


@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_mnist_gpu(framework_version, ecr_image, sagemaker_regions, dist_gpu_backend):
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            'entry_point': mnist_script,
            'role': 'SageMakerRole',
            'instance_count': 2,
            'framework_version': framework_version,
            'instance_type': MULTI_GPU_INSTANCE,
            'hyperparameters': {'backend': dist_gpu_backend},
        }
        upload_s3_data_args = {
        'path': os.path.join(data_dir, 'training'),
        'key_prefix': 'pytorch/mnist'
        }
        job_name=utils.unique_name_from_base('test-pt-mnist-gpu')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, upload_s3_data_args=upload_s3_data_args, job_name=job_name)



@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_pt_mnist.py", 8)])
def test_smmodelparallel_mnist_multigpu_multinode(ecr_image, instance_type, sagemaker_regions, test_script, num_processes):
    """
    Tests pt mnist command via script mode
    """
    instance_type = "ml.p3.16xlarge"
    validate_or_skip_smmodelparallel(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            'entry_point': test_script,
            'role': 'SageMakerRole',
            'source_dir': mnist_path,
            'instance_count': 2,
            'instance_type': instance_type,
            'hyperparameters': {"assert-losses": 1, "amp": 1, "ddp": 1, "data-dir": "data/training", "epochs": 5},
            'distribution': {
                "smdistributed": {
                    "modelparallel": {
                        "enabled": True,
                        "parameters": {
                            "partitions": 2,
                            "microbatches": 4,
                            "optimize": "speed",
                            "pipeline": "interleaved",
                            "ddp": True,
                        },
                    }
                },
                "mpi": {
                    "enabled": True,
                    "processes_per_host": num_processes,
                    "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x SMDEBUG_LOG_LEVEL=error -x OMPI_MCA_btl_vader_single_copy_mechanism=none ",
                },
            },
        }
        job_name=utils.unique_name_from_base('test-pt-smdmp-multinode')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)

@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_pt_mnist.py", 8)])
@pytest.mark.efa()
def test_smmodelparallel_mnist_multigpu_multinode_efa(ecr_image, efa_instance_type, sagemaker_regions, test_script, num_processes):
    """
    Tests pt mnist command via script mode
    """
    validate_or_skip_smmodelparallel_efa(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):

        estimator_parameter = {
            'entry_point': test_script,
            'role': 'SageMakerRole',
            'source_dir': mnist_path,
            'instance_count': 2,
            'instance_type': efa_instance_type,
            'hyperparameters': {"assert-losses": 1, "amp": 1, "ddp": 1, "data-dir": "data/training", "epochs": 5},
            'distribution': {
                "smdistributed": {
                    "modelparallel": {
                        "enabled": True,
                        "parameters": {
                            "partitions": 2,
                            "microbatches": 4,
                            "optimize": "speed",
                            "pipeline": "interleaved",
                            "ddp": True,
                        },
                    }
                },
                "mpi": {
                    "enabled": True,
                    "processes_per_host": num_processes,
                    "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x SMDEBUG_LOG_LEVEL=error -x OMPI_MCA_btl_vader_single_copy_mechanism=none -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa ",
                },
        }
        }
        job_name=utils.unique_name_from_base('test-pt-smdmp-multinode-efa')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)


@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.efa()
@pytest.mark.skip_py2_containers
def test_sanity_efa(ecr_image, efa_instance_type, sagemaker_regions):
    """
    Tests pt mnist command via script mode
    """
    validate_or_skip_smmodelparallel_efa(ecr_image)
    efa_test_path = os.path.join(RESOURCE_PATH, 'efa', 'test_efa.sh')
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            'entry_point': efa_test_path,
            'role': 'SageMakerRole',
            'instance_count': 1,
            'instance_type': efa_instance_type,
            'distribution': {
                "mpi": {
                    "enabled": True,
                    "processes_per_host": 1
                },
            },
        }
        job_name=utils.unique_name_from_base('test-pt-efa-sanity')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)


def _test_dist_operations(
        ecr_image, sagemaker_session, framework_version, instance_type, dist_backend, instance_count=3
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
        pytorch.fit({'required_argument': fake_input}, job_name=utils.unique_name_from_base('test-pt-dist-operations'))


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
