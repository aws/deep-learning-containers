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
import tarfile
from pathlib import Path
from urllib.parse import urlparse
import pytest
import sagemaker
from sagemaker import utils
from sagemaker.huggingface import HuggingFace
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from retrying import retry
from sagemaker import get_execution_role
from packaging.specifiers import SpecifierSet

from ..... import invoke_sm_helper_function

# configuration for running training on torch distributed Data Parallel
distribution = {"torch_distributed": {"enabled": True}}

# hyperparameters, which are passed into the training job
hyperparameters = {
    "model_id": "Qwen/Qwen3-8B",
    "tensor_parallel_size": 8,
    "gradient_accumulation_steps": 8,
    "zero_1": True,
    "do_train": True,
    "bf16": True,
    "max_steps": 1,
    "per_device_train_batch_size": 1,
    "num_train_epochs": 1,
    "logging_steps": 1,
    "output_dir": "/opt/ml/model",
    "report_to": "none",
}


# ValueError: Must setup local AWS configuration with a region supported by SageMaker.
def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


def get_pytorch_version(ecr_image):
    pytorch_version_search = re.search(r"(\d+(\.\d+){1,2})-transformers", ecr_image)
    if pytorch_version_search:
        pytorch_version = pytorch_version_search.group(1)
        return pytorch_version
    else:
        raise LookupError("PyTorch version not found in image URI")


# TBD. This function is mainly there to handle capacity issues now. Once trn1 capacaity issues
# are fixed, we can remove this function
@retry(
    stop_max_attempt_number=8,
    wait_fixed=20000,
    retry_on_exception=retry_if_value_error,
)
def invoke_neuron_helper_function(
    ecr_image, sagemaker_regions, helper_function, helper_function_args
):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and the sagemaker
    session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues are seen.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param helper_function: Function to invoke
    :param helper_function_args: Helper function args

    :return: None
    """
    from ..... import get_ecr_image_region, get_sagemaker_session, get_ecr_image

    ecr_image_region = get_ecr_image_region(ecr_image)
    for region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        )
        try:
            helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                raise ValueError("CapacityError: Retry.")
            else:
                raise e


@pytest.mark.processor("neuronx")
@pytest.mark.model("hf-pt-glue-neuronx")
@pytest.mark.neuronx_test
@pytest.mark.skip_py2_containers
@pytest.mark.team("sagemaker-1p-algorithms")
def test_neuronx_text_classification(ecr_image, sagemaker_regions, py_version, instance_type):
    if get_pytorch_version(ecr_image) in SpecifierSet("==2.14.*"):
        pytest.skip("Native Torch NeuronX is covered by test_neuronx_native_training")
    function_args = {
        "py_version": py_version,
        "instance_type": instance_type,
        "instance_count": 1,
        "num_neuron_cores": 2,
    }

    invoke_neuron_helper_function(
        ecr_image, sagemaker_regions, _test_neuronx_text_classification_function, function_args
    )


def _test_neuronx_text_classification_function(
    ecr_image,
    sagemaker_session,
    py_version,
    instance_type="ml.trn1.32xlarge",
    instance_count=1,
    num_neuron_cores=2,
):
    pytorch_version = get_pytorch_version(ecr_image)
    if pytorch_version in SpecifierSet("==2.7.*"):
        optimum_neuron_version = "0.3.0"
    elif pytorch_version in SpecifierSet("==2.8.*"):
        optimum_neuron_version = "0.4.1"
    else:
        raise ValueError(
            f"`optimum_neuron_version` to be set for PyTorch version {pytorch_version}."
        )

    git_config = {
        "repo": "https://github.com/huggingface/optimum-neuron.git",
        "branch": "v" + optimum_neuron_version,
    }

    source_dir = "./examples/training/qwen3"

    role = get_execution_role()
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator = HuggingFace(
            entry_point="finetune_qwen3.py",
            source_dir=source_dir,
            git_config=git_config,
            role="SageMakerRole",
            image_uri=ecr_image,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            py_version=py_version,
            distribution=distribution,
            hyperparameters=hyperparameters,
        )
        estimator.fit(job_name=sagemaker.utils.unique_name_from_base("test-hf-pt-qwen3-neuronx"))


@pytest.mark.processor("neuronx")
@pytest.mark.model("qwen3")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.parametrize("parallel_mode", ["tp", "fsdp"])
def test_neuronx_native_training(
    ecr_image, sagemaker_regions, py_version, instance_type, parallel_mode, tmp_path
):
    if get_pytorch_version(ecr_image) not in SpecifierSet("==2.14.*"):
        pytest.skip("Requires native Torch NeuronX 2.14")
    invoke_sm_helper_function(
        ecr_image,
        sagemaker_regions,
        _test_neuronx_native_training,
        py_version,
        instance_type,
        parallel_mode,
        tmp_path,
    )


def _test_neuronx_native_training(
    ecr_image, sagemaker_session, py_version, instance_type, parallel_mode, tmp_path
):
    # Reuse the EC2 worker to exercise the same TP/FSDP assertions on SageMaker.
    source_dir = Path(__file__).resolve().parents[5] / "dlc_tests/container_tests/bin/huggingface"
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator = HuggingFace(
            entry_point="transformers_neuron_training.py",
            source_dir=str(source_dir),
            role="SageMakerRole",
            image_uri=ecr_image,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            py_version=py_version,
            distribution={"torch_distributed": {"enabled": True}},
            hyperparameters={
                "parallel-mode": parallel_mode,
                "sagemaker_mpi_num_of_processes_per_host": 2,
            },
            environment={
                "ON_NEURON_EAGER": "1",
                "HF_DEACTIVATE_ASYNC_LOAD": "1",
                "OMP_NUM_THREADS": "4",
                "TORCH_NEURONX_ENABLE_HOST_CC": "1",
                "TORCH_NEURONX_ENABLE_ASYNC_NRT": "1",
            },
        )
        estimator.fit(
            job_name=sagemaker.utils.unique_name_from_base(f"test-hf-neuronx-{parallel_mode}")
        )
        # Verify SageMaker exported the trained model from /opt/ml/model to S3.
        artifact = urlparse(estimator.model_data)
        local_model = tmp_path / "model.tar.gz"
        sagemaker_session.boto_session.client("s3").download_file(
            artifact.netloc, artifact.path.lstrip("/"), str(local_model)
        )
        with tarfile.open(local_model) as archive:
            names = {name.removeprefix("./") for name in archive.getnames()}
        assert {"config.json", "model.safetensors"} <= names, names
