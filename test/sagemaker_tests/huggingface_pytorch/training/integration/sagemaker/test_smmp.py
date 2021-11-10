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
import pytest
from ...integration import DEFAULT_TIMEOUT
from sagemaker.huggingface import HuggingFace
from ...integration.sagemaker.timeout import timeout
import sagemaker

# hyperparameters, which are passed into the training job
hyperparameters = {
    "model_name_or_path": "roberta-large",
    "task_name": "mnli",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "do_train": True,
    "do_eval": True,
    "do_predict": True,
    "output_dir": "/opt/ml/model",
    "max_steps": 500,
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


def get_transformers_version(ecr_image):
    transformers_version_search = re.search(r"transformers(\d+(\.\d+){1,2})", ecr_image)
    if transformers_version_search:
        transformers_version = transformers_version_search.group(1)
        return transformers_version
    else:
        raise LookupError("HF transformers version not found in image URI")


@pytest.mark.processor("gpu")
@pytest.mark.integration("smmp")
@pytest.mark.model("hf_qa_smmp")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
def test_smmp_gpu(sagemaker_session, framework_version, ecr_image, instance_type, py_version, dist_gpu_backend):
    # instance configurations
    instance_type = "ml.p3.16xlarge"
    instance_count = 1
    volume_size = 400

    transformers_version = get_transformers_version(ecr_image)
    git_config = {"repo": "https://github.com/huggingface/transformers.git", "branch": "v" + transformers_version}

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


@pytest.mark.processor("gpu")
@pytest.mark.integration("smmp")
@pytest.mark.model("hf_qa_smmp_multi")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.multinode(2)
@pytest.mark.skip_trcomp_containers
def test_smmp_gpu_multinode(
    sagemaker_session, framework_version, ecr_image, instance_type, py_version, dist_gpu_backend
):
    instance_type = "ml.p3.16xlarge"
    instance_count = 2
    volume_size = 400

    transformers_version = get_transformers_version(ecr_image)
    git_config = {"repo": "https://github.com/huggingface/transformers.git", "branch": "v" + transformers_version}

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
    huggingface_estimator.fit(job_name=sagemaker.utils.unique_name_from_base("test-hf-pt-qa-smmp-multi"))
