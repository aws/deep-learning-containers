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
"""
SageMaker SDK v3 Training Utilities

This module provides v3-native utilities for PyTorch training tests using ModelTrainer.
"""
from __future__ import absolute_import

import botocore.exceptions
import pytest

try:
    from sagemaker.exceptions import UnexpectedStatusException
except (ImportError, ModuleNotFoundError):
    from sagemaker.core.exceptions import UnexpectedStatusException

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.train.distributed import Torchrun
from sagemaker import utils
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_delay

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


def upload_s3_data(sagemaker_session, path, key_prefix):
    """Upload data to S3 for training."""
    sagemaker_session.default_bucket()
    return sagemaker_session.upload_data(path=path, key_prefix=key_prefix)


def create_source_code(entry_script, source_dir=None, dependencies=None):
    """Create v3 SourceCode config."""
    return SourceCode(
        entry_script=entry_script,
        source_dir=source_dir,
        dependencies=dependencies,
    )


def create_compute(instance_type, instance_count=1, volume_size=30, keep_alive_seconds=0):
    """Create v3 Compute config."""
    return Compute(
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size,
        keep_alive_period_in_seconds=keep_alive_seconds,
    )


def create_input_data(channel_name, data_source):
    """Create v3 InputData config."""
    return InputData(channel_name=channel_name, data_source=data_source)


def get_distributed_runner(dist_type):
    """
    Get v3 distributed runner.

    In SDK v3, SMDataParallel is no longer available as a separate class.
    Use Torchrun for all distributed training scenarios.

    :param dist_type: One of 'torchrun', 'smddp', or None
    :return: Torchrun or None
    """
    if dist_type in ("torchrun", "smddp"):
        # In v3, both torchrun and smddp use Torchrun distributed runner
        # SMDDP functionality is handled at the container/script level
        return Torchrun()
    return None


@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (SMInstanceCapacityError, SMThrottlingError, SMResourceLimitExceededError)
    ),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_pytorch_training(
    ecr_image,
    sagemaker_regions,
    source_code,
    compute,
    hyperparameters=None,
    input_data_config=None,
    distributed_runner=None,
    environment=None,
    role="SageMakerRole",
    job_name=None,
    upload_s3_data_args=None,
):
    """
    Invoke PyTorch training job using SageMaker SDK v3 ModelTrainer.

    :param ecr_image: ECR image URI
    :param sagemaker_regions: List of SageMaker regions to try
    :param source_code: v3 SourceCode config
    :param compute: v3 Compute config
    :param hyperparameters: Dict of hyperparameters
    :param input_data_config: List of v3 InputData configs
    :param distributed_runner: v3 distributed runner (Torchrun or SMDataParallel)
    :param environment: Dict of environment variables
    :param role: IAM role name
    :param job_name: Base job name
    :param upload_s3_data_args: Dict with 'path' and 'key_prefix' for S3 upload
    :return: tuple (ModelTrainer, sagemaker_session)
    """
    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None

    for test_region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(test_region)
        tested_ecr_image = (
            get_ecr_image(ecr_image, test_region) if test_region != ecr_image_region else ecr_image
        )

        env = environment.copy() if environment else {}
        env["AWS_REGION"] = test_region

        try:
            model_trainer = ModelTrainer(
                training_image=tested_ecr_image,
                source_code=source_code,
                compute=compute,
                hyperparameters=hyperparameters or {},
                role=role,
                sagemaker_session=sagemaker_session,
                base_job_name=job_name,
                distributed_runner=distributed_runner,
                environment=env,
            )

            # Handle data upload if specified
            final_input_config = input_data_config or []
            if upload_s3_data_args:
                training_input = upload_s3_data(sagemaker_session, **upload_s3_data_args)
                final_input_config.append(
                    InputData(channel_name="training", data_source=training_input)
                )

            # Generate unique job name
            unique_job_name = utils.unique_name_from_base(job_name) if job_name else None

            # Start training
            model_trainer.train(
                input_data_config=final_input_config if final_input_config else None,
                job_name=unique_job_name,
                wait=True,
            )
            return model_trainer, sagemaker_session

        except UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                error = e
                continue
            raise e
        except botocore.exceptions.ClientError as e:
            if any(ex in str(e) for ex in ["ThrottlingException", "ResourceLimitExceeded"]):
                error = e
                continue
            raise e

    # Handle failures
    instance_type = compute.instance_type
    if instance_type in LOW_AVAILABILITY_INSTANCE_TYPES:
        pytest.skip(f"Failed to launch job due to low capacity on {instance_type}")

    if "CapacityError" in str(error):
        raise SMInstanceCapacityError from error
    elif "ResourceLimitExceeded" in str(error):
        raise SMResourceLimitExceededError from error
    else:
        raise SMThrottlingError from error


def _test_mnist_distributed(
    ecr_image,
    sagemaker_session,
    framework_version,
    dist_backend,
    instance_type=None,
    instance_groups=None,
    use_inductor=False,
):
    """Test MNIST distributed training using v3 ModelTrainer."""

    # In SDK v3, use Torchrun for all distributed training
    # The backend (nccl/gloo) is specified via hyperparameters
    distributed_runner = Torchrun()

    # Build v3 configs
    source_code = create_source_code(
        entry_script=mnist_script.split("/")[-1] if "/" in mnist_script else mnist_script,
        source_dir=training_dir,
    )

    # Determine instance settings
    if instance_groups:
        inst_type = instance_groups[0].instance_type
        inst_count = instance_groups[0].instance_count
        job_name = "test-pt-hc-mnist-distributed"
    else:
        inst_type = instance_type
        inst_count = 2
        job_name = "test-pt-mnist-distributed"

    compute = create_compute(instance_type=inst_type, instance_count=inst_count)

    hyperparameters = {
        "backend": dist_backend,
        "epochs": 1,
        "inductor": int(use_inductor),
    }

    with timeout(minutes=DEFAULT_TIMEOUT):
        model_trainer = ModelTrainer(
            training_image=ecr_image,
            source_code=source_code,
            compute=compute,
            hyperparameters=hyperparameters,
            role="SageMakerRole",
            sagemaker_session=sagemaker_session,
            distributed_runner=distributed_runner,
        )

        # Upload training data
        training_input = sagemaker_session.upload_data(
            path=training_dir, key_prefix="pytorch/mnist"
        )

        input_data = create_input_data(channel_name="training", data_source=training_input)

        model_trainer.train(
            input_data_config=[input_data],
            job_name=utils.unique_name_from_base(job_name),
            wait=True,
        )
