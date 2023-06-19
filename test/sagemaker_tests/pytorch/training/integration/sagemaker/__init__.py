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

import pytest
import sagemaker

from sagemaker.pytorch import PyTorch
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
    SMThrottlingError,
)


def upload_s3_data(estimator, path, key_prefix):
    estimator.sagemaker_session.default_bucket()
    inputs = estimator.sagemaker_session.upload_data(path=path, key_prefix=key_prefix)
    return inputs


@retry(
    reraise=True,
    retry=retry_if_exception_type((SMInstanceCapacityError, SMThrottlingError)),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_pytorch_estimator(
    ecr_image,
    sagemaker_regions,
    estimator_parameter,
    inputs=None,
    disable_sm_profiler=False,
    upload_s3_data_args=None,
    job_name=None,
):
    """
    Used to invoke PyTorch training job. The ECR image and the sagemaker session are used depending
    on the AWS region. This function will rerun for all SM regions after a defined wait time if
    capacity issues occur.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param estimator_parameter: Estimator parameters for SM job.
    :param inputs: Inputs for fit estimator call
    :param disable_sm_profiler: Flag to disable SM profiler
    :param upload_s3_data_args: Data to be uploded to S3 for training job
    :param job_name: Training job name

    :return: None
    """

    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None
    for test_region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(test_region)
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, test_region) if test_region != ecr_image_region else ecr_image
        )
        if "environment" not in estimator_parameter:
            estimator_parameter["environment"] = {"AWS_REGION": test_region}
        else:
            estimator_parameter["environment"]["AWS_REGION"] = test_region
        try:
            pytorch = PyTorch(
                image_uri=tested_ecr_image,
                sagemaker_session=sagemaker_session,
                **estimator_parameter,
            )

            if disable_sm_profiler:
                if sagemaker_session.boto_region_name in ("cn-north-1", "cn-northwest-1"):
                    pytorch.disable_profiler = True

            if upload_s3_data_args:
                training_input = upload_s3_data(pytorch, **upload_s3_data_args)
                inputs = {"training": training_input}

            pytorch.fit(inputs=inputs, job_name=job_name, logs=True)
            return pytorch, sagemaker_session

        except sagemaker.exceptions.UnexpectedStatusException as e:
            error = e
            if "CapacityError" in str(e):
                time.sleep(0.5)
                continue
            elif "ThrottlingException" in str(e):
                time.sleep(5)
                continue
            else:
                raise e

    instance_types = []
    if "instance_type" in estimator_parameter:
        instance_types = [estimator_parameter["instance_type"]]
    elif "instance_groups" in estimator_parameter:
        instance_types = [
            instance_group.instance_type
            for instance_group in estimator_parameter["instance_groups"]
        ]
    # It is possible to have such low capacity on certain instance types that the test is never able
    # to run due to ICE errors. In these cases, we are forced to xfail/skip the test, or end up
    # causing pipelines to fail forever. We have approval to skip the test when this type of ICE
    # error occurs for p4de. Will need approval for each new instance type to be added to this list.
    if any(instance_type in LOW_AVAILABILITY_INSTANCE_TYPES for instance_type in instance_types):
        # TODO: xfailed tests do not show up on CodeBuild Test Case Reports. Therefore using "skip"
        #       instead of xfail.
        pytest.skip(f"Failed to launch job due to low capacity on {instance_types}")
    if "CapacityError" in str(error):
        raise SMInstanceCapacityError from error
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
    dist_method = "pytorchddp" if dist_backend.lower() == "nccl" else "torch_distributed"
    est_params = {
        "entry_point": mnist_script,
        "role": "SageMakerRole",
        "sagemaker_session": sagemaker_session,
        "image_uri": ecr_image,
        "hyperparameters": {"backend": dist_backend, "epochs": 1, "inductor": int(use_inductor)},
        "framework_version": framework_version,
        "distribution": {dist_method: {"enabled": True}},
    }
    if not instance_groups:
        est_params["instance_type"] = instance_type
        est_params["instance_count"] = 2
    else:
        est_params["instance_groups"] = instance_groups
    job_name = "test-pt-hc-mnist-distributed" if instance_groups else "test-pt-mnist-distributed"
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(**est_params)
        training_input = pytorch.sagemaker_session.upload_data(
            path=training_dir, key_prefix="pytorch/mnist"
        )
        pytorch.fit({"training": training_input}, job_name=utils.unique_name_from_base(job_name))
