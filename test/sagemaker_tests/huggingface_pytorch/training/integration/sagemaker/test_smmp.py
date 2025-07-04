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
import re
from distutils.version import Version
from test.test_utils import get_cuda_version_from_tag, get_framework_and_version_from_tag

import pytest
import sagemaker
from sagemaker.huggingface import HuggingFace

from ..... import invoke_sm_helper_function
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout

# hyperparameters, which are passed into the training job
hyperparameters = {
    "model_name_or_path": "hf-internal-testing/tiny-random-RobertaModel",
    "task_name": "mnli",
    "per_device_train_batch_size": 2,  # batch size must be divisible by the number of microbatches
    "per_device_eval_batch_size": 2,
    "do_train": True,
    "do_eval": True,
    "do_predict": True,
    "output_dir": "/opt/ml/model",
    "max_steps": 10,
    "max_train_samples": 30,
}

# configuration for running training on smdistributed Model Parallel
mpi_options = {
    "enabled": True,
    "processes_per_host": 8,
}
smp_options = {
    "enabled": True,
    "parameters": {
        "microbatches": 2,
        "placement_strategy": "spread",
        "pipeline": "interleaved",
        "optimize": "speed",
        "partitions": 2,
        "ddp": True,
    },
}

distribution = {"smdistributed": {"modelparallel": smp_options}, "mpi": mpi_options}


def get_transformers_version_from_image_uri(ecr_image):
    transformers_version_search = re.search(r"transformers(\d+(\.\d+){1,2})", ecr_image)
    if transformers_version_search:
        transformers_version = transformers_version_search.group(1)
        return transformers_version
    else:
        raise LookupError("HF transformers version not found in image URI")


def validate_or_skip_modelparallel(ecr_image):
    if not can_run_modelparallel(ecr_image):
        pytest.skip("Model Parallelism is supported on CUDA 11 with PyTorch < v2.0")


def can_run_modelparallel(ecr_image):
    image_framework, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)

    # Check if the framework is PyTorch
    if image_framework.lower() != "pytorch":
        return False

    # Convert versions to appropriate formats
    framework_version = Version(image_framework_version)
    cuda_version = Version(image_cuda_version.strip("cu"))

    return (framework_version < Version("2.1")) and (cuda_version == Version("110"))


@pytest.mark.processor("gpu")
@pytest.mark.integration("smmp")
@pytest.mark.model("hf_qa_smmp")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.team("sagemaker-1p-algorithms")
def test_smmp_gpu(
    ecr_image,
    sagemaker_regions,
    instance_type,
    framework_version,
    py_version,
):
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_smmp_gpu_function, py_version, 1)


@pytest.mark.processor("gpu")
@pytest.mark.integration("smmp")
@pytest.mark.model("hf_qa_smmp_multi")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.multinode(2)
@pytest.mark.skip_trcomp_containers
@pytest.mark.team("sagemaker-1p-algorithms")
def test_smmp_gpu_multinode(
    ecr_image,
    sagemaker_regions,
    instance_type,
    framework_version,
    py_version,
):
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_smmp_gpu_function, py_version, 2)


def _test_smmp_gpu_function(ecr_image, sagemaker_session, py_version, instances_quantity):
    instance_type = "ml.g5.48xlarge"
    instance_count = instances_quantity
    volume_size = 400

    transformers_version = get_transformers_version_from_image_uri(ecr_image)
    git_config = {
        "repo": "https://github.com/huggingface/transformers.git",
        "branch": "v" + transformers_version,
    }

    validate_or_skip_modelparallel(ecr_image)

    huggingface_estimator = HuggingFace(
        entry_point="run_glue.py",
        source_dir="./examples/pytorch/text-classification",
        git_config=git_config,
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size=volume_size,
        role="SageMakerRole",
        image_uri=ecr_image,
        distribution=distribution,
        py_version=py_version,
        hyperparameters=hyperparameters,
        sagemaker_session=sagemaker_session,
    )
    huggingface_estimator.fit(job_name=sagemaker.utils.unique_name_from_base("test-hf-pt-qa-smmp"))
