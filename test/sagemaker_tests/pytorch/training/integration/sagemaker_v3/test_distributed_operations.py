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
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, Compute
from urllib.parse import urlparse
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ....training import get_efa_test_instance_type
from ...integration import (
    data_dir,
    dist_operations_path,
    fastai_path,
    mnist_script,
    DEFAULT_TIMEOUT,
    mnist_path,
    gpt2_path,
)
from .timeout import timeout
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer

MULTI_GPU_INSTANCE = "ml.g5.12xlarge"
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


def validate_or_skip_smmodelparallel(ecr_image):
    if not can_run_smmodelparallel(ecr_image):
        pytest.skip("Model Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smmodelparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


def validate_or_skip_smmodelparallel_efa(ecr_image):
    if not can_run_smmodelparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on PyTorch 1.8.1 or higher")


def skip_unsupported_instances_smmodelparallel(instance_type):
    if instance_type.startswith("ml.p5"):
        pytest.skip(f"{instance_type} is not supported by smdataparallel")


def can_run_smmodelparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.8.1") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


@pytest.mark.processor("cpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
@pytest.mark.deploy_test
@pytest.mark.skip_test_in_region
@pytest.mark.team("conda")
def test_dist_operations_cpu(
    framework_version, ecr_image, sagemaker_regions, instance_type, dist_cpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.c5.xlarge"

    source_code = SourceCode(
        source_dir=os.path.dirname(dist_operations_path),
        entry_script=os.path.basename(dist_operations_path),
    )
    compute_params = {"instance_type": instance_type, "instance_count": 3}
    hyperparameters = {"backend": dist_cpu_backend}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-dist-operations",
        )


@pytest.mark.processor("gpu")
@pytest.mark.multinode(3)
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
@pytest.mark.deploy_test
@pytest.mark.team("conda")
def test_dist_operations_gpu(
    framework_version, instance_type, ecr_image, sagemaker_regions, dist_gpu_backend
):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.g5.4xlarge"

    source_code = SourceCode(
        source_dir=os.path.dirname(dist_operations_path),
        entry_script=os.path.basename(dist_operations_path),
    )
    compute_params = {"instance_type": instance_type, "instance_count": 3}
    hyperparameters = {"backend": dist_gpu_backend}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-dist-operations",
        )


@pytest.mark.processor("gpu")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
@pytest.mark.team("conda")
def test_dist_operations_multi_gpu(
    framework_version, ecr_image, sagemaker_regions, dist_gpu_backend
):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=os.path.dirname(dist_operations_path),
        entry_script=os.path.basename(dist_operations_path),
    )
    compute_params = {"instance_type": MULTI_GPU_INSTANCE, "instance_count": 1}
    hyperparameters = {"backend": dist_gpu_backend}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-dist-operations-multigpu",
        )


@pytest.mark.processor("gpu")
@pytest.mark.integration("fastai")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.team("conda")
def test_dist_operations_fastai_gpu(framework_version, ecr_image, sagemaker_regions):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=fastai_path,
        entry_script="train_distributed.py",
    )
    compute_params = {"instance_type": MULTI_GPU_INSTANCE, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-fastai",
        )


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("gpt2")
@pytest.mark.processor("gpu")
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("train_gpt_simple.py", 8)])
def test_smmodelparallel_gpt2_multigpu_singlenode(
    ecr_image, instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    # TODO: Implement v3 equivalent for smmodelparallel tests when needed
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("gpt2")
@pytest.mark.processor("gpu")
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("train_gpt_simple.py", 8)])
def test_smmodelparallel_gpt2_multigpu_singlenode_flashattn(
    ecr_image, instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_pt_mnist.py", 8)])
def test_smmodelparallel_mnist_multigpu_multinode(
    ecr_image, instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_pt_mnist.py", 8)])
def test_hc_smmodelparallel_mnist_multigpu_multinode(
    ecr_image, instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_pt_mnist.py", 8)])
@pytest.mark.efa()
def test_smmodelparallel_mnist_multigpu_multinode_efa(
    ecr_image, efa_instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip("SM Model Parallel team is maintaining their own Docker Container")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("gpt2")
@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.team("smmodelparallel")
@pytest.mark.parametrize("test_script, num_processes", [("train_gpt_simple.py", 8)])
@pytest.mark.efa()
def test_smmodelparallel_gpt2_sdp_multinode_efa(
    ecr_image, efa_instance_type, sagemaker_regions, test_script, num_processes
):
    skip_if_not_v3_compatible(ecr_image)
    pytest.skip("SM Model Parallel v3 test not yet implemented")


@pytest.mark.skip(reason="Sagemaker efa test is a duplicate of ec2 efa test on p4d instances")
@pytest.mark.integration("smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.efa()
@pytest.mark.skip_py2_containers
@pytest.mark.team("conda")
def test_sanity_efa(ecr_image, efa_instance_type, sagemaker_regions):
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smmodelparallel_efa(ecr_image)
    skip_unsupported_instances_smmodelparallel(efa_instance_type)
    efa_test_path = os.path.join(RESOURCE_PATH, "efa", "test_efa.sh")

    source_code = SourceCode(
        source_dir=os.path.dirname(efa_test_path),
        entry_script=os.path.basename(efa_test_path),
    )
    compute_params = {"instance_type": efa_instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-efa-sanity",
        )
