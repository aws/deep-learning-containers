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
import pytest
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute
from sagemaker.modules.distributed import Torchrun
from ...integration import neuron_allreduce_path, neuron_mlp_path, DEFAULT_TIMEOUT
from .timeout import timeout
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer


@pytest.mark.processor("neuronx")
@pytest.mark.model("unknown_model")
@pytest.mark.parametrize("instance_types", ["ml.trn1.32xlarge"])
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_allreduce_distributed(
    framework_version, ecr_image, sagemaker_regions, instance_types
):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=neuron_allreduce_path,
        entry_script="all_reduce.py",
    )
    compute_params = {"instance_type": instance_types, "instance_count": 2}
    distributed_runner = Torchrun()

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            distributed_runner=distributed_runner,
            environment={"FI_EFA_FORK_SAFE": "1"},
            job_name="test-pt-v3-neuron-allreduce-dist",
        )


@pytest.mark.processor("neuronx")
@pytest.mark.model("mlp")
@pytest.mark.parametrize("instance_types", ["ml.trn1.32xlarge"])
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_mlp_distributed(framework_version, ecr_image, sagemaker_regions, instance_types):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=neuron_mlp_path,
        entry_script="train_torchrun.py",
    )
    compute_params = {"instance_type": instance_types, "instance_count": 2}
    distributed_runner = Torchrun()

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            distributed_runner=distributed_runner,
            environment={"FI_EFA_FORK_SAFE": "1"},
            job_name="test-pt-v3-neuron-mlp-dist",
        )


@pytest.mark.processor("neuronx")
@pytest.mark.model("unknown_model")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_allreduce_process(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=neuron_allreduce_path,
        entry_script="entrypoint.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}
    hyperparameters = {"nproc-per-node": 2, "nnodes": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            upload_s3_data_args={
                "path": neuron_allreduce_path,
                "key_prefix": "pytorch/neuron_allreduce",
            },
            job_name="test-pt-v3-neuron-allreduce",
        )


@pytest.mark.processor("neuronx")
@pytest.mark.model("mlp")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_mlp_process(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)

    source_code = SourceCode(
        source_dir=neuron_mlp_path,
        entry_script="entrypoint.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}
    hyperparameters = {"nproc-per-node": 2, "nnodes": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            upload_s3_data_args={
                "path": neuron_mlp_path,
                "key_prefix": "pytorch/neuron_mlp",
            },
            job_name="test-pt-v3-neuron-mlp",
        )
