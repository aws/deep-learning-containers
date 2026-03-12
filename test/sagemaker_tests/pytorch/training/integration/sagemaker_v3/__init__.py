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

import time

import boto3
import botocore.exceptions
import pytest

from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute
from sagemaker.modules.distributed import Torchrun
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_delay

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from test.test_utils import get_framework_and_version_from_tag

from .timeout import timeout
from ...integration import training_dir, mnist_script, DEFAULT_TIMEOUT
from ..... import (
    get_ecr_image,
    get_ecr_image_region,
    get_sagemaker_session,
    LOW_AVAILABILITY_INSTANCE_TYPES,
    SMInstanceCapacityError,
    SMResourceLimitExceededError,
    SMThrottlingError,
)


def skip_if_not_v3_compatible(ecr_image):
    """Skip test if the image is not PyTorch >= 2.10 (v3 SDK only)."""
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    if Version(image_framework_version) not in SpecifierSet(">=2.10"):
        pytest.skip("SageMaker SDK v3 tests only run on PyTorch >= 2.10 images")


def upload_s3_data_v3(sagemaker_session, path, key_prefix):
    sagemaker_session.default_bucket()
    inputs = sagemaker_session.upload_data(path=path, key_prefix=key_prefix)
    return inputs


@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (SMInstanceCapacityError, SMThrottlingError, SMResourceLimitExceededError)
    ),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_pytorch_model_trainer(
    ecr_image,
    sagemaker_regions,
    source_code,
    compute_params,
    hyperparameters=None,
    distributed_runner=None,
    input_data_config=None,
    upload_s3_data_args=None,
    job_name=None,
    environment=None,
):
    """
    Used to invoke PyTorch training job using SageMaker SDK v3 ModelTrainer.
    The ECR image and the sagemaker session are used depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if
    capacity issues occur.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param source_code: SourceCode config for ModelTrainer
    :param compute_params: dict with instance_type, instance_count
    :param hyperparameters: dict of hyperparameters
    :param distributed_runner: Torchrun or other distributed config
    :param input_data_config: list of InputData objects
    :param upload_s3_data_args: Data to be uploaded to S3 for training job
    :param job_name: Training job base name
    :param environment: dict of environment variables

    :return: (model_trainer, sagemaker_session)
    """

    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None
    for test_region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(test_region)
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, test_region) if test_region != ecr_image_region else ecr_image
        )

        env = environment.copy() if environment else {}
        env["AWS_REGION"] = test_region

        try:
            compute = Compute(
                instance_type=compute_params.get("instance_type", "ml.m5.xlarge"),
                instance_count=compute_params.get("instance_count", 1),
            )

            trainer_kwargs = {
                "training_image": tested_ecr_image,
                "source_code": source_code,
                "compute": compute,
            }
            if hyperparameters:
                trainer_kwargs["hyperparameters"] = hyperparameters
            if distributed_runner:
                trainer_kwargs["distributed_runner"] = distributed_runner
            if job_name:
                trainer_kwargs["base_job_name"] = job_name
            if env:
                trainer_kwargs["environment"] = env

            model_trainer = ModelTrainer(**trainer_kwargs)

            if upload_s3_data_args:
                training_input = upload_s3_data_v3(
                    sagemaker_session,
                    upload_s3_data_args["path"],
                    upload_s3_data_args["key_prefix"],
                )
                input_data_config = [InputData(channel_name="training", data_source=training_input)]

            model_trainer.train(
                input_data_config=input_data_config,
                wait=True,
            )
            return model_trainer, sagemaker_session

        except Exception as e:
            error_str = str(e)
            if "CapacityError" in error_str:
                error = e
                continue
            elif any(
                exc_type in error_str
                for exc_type in ["ThrottlingException", "ResourceLimitExceeded"]
            ):
                error = e
                continue
            else:
                raise e

    instance_types = []
    if "instance_type" in compute_params:
        instance_types = [compute_params["instance_type"]]
    if any(instance_type in LOW_AVAILABILITY_INSTANCE_TYPES for instance_type in instance_types):
        pytest.skip(f"Failed to launch job due to low capacity on {instance_types}")
    if error and "CapacityError" in str(error):
        raise SMInstanceCapacityError from error
    elif error and "ResourceLimitExceeded" in str(error):
        raise SMResourceLimitExceededError from error
    else:
        raise SMThrottlingError from error


def _test_mnist_distributed_v3(
    ecr_image,
    sagemaker_regions,
    framework_version,
    dist_backend,
    instance_type=None,
    instance_count=2,
    use_inductor=False,
):
    """v3 equivalent of _test_mnist_distributed using ModelTrainer."""
    from ...integration import mnist_path, mnist_script

    hyperparameters = {"backend": dist_backend, "epochs": 1}
    if use_inductor:
        hyperparameters["inductor"] = 1

    source_code = SourceCode(
        source_dir=mnist_path,
        entry_script="mnist.py",
    )

    compute_params = {
        "instance_type": instance_type or "ml.m5.xlarge",
        "instance_count": instance_count,
    }

    distributed_runner = Torchrun() if dist_backend.lower() in ("nccl", "gloo") else None

    job_name = "test-pt-v3-mnist-distributed"
    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            distributed_runner=distributed_runner,
            job_name=job_name,
        )
