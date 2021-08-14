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
from .... import get_ecr_image, get_ecr_image_region, get_sagemaker_session
import sagemaker
from sagemaker.pytorch import PyTorch
import time

def upload_s3_data(estimator, path, key_prefix):

    estimator.sagemaker_session.default_bucket()
    inputs = estimator.sagemaker_session.upload_data(
        path=path,
        key_prefix=key_prefix)
    return inputs


def invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, inputs=None, disable_sm_profiler=False, upload_s3_data_args=None, job_name=None):
    """
    Used to invoke PyTorch training job. The ECR image and the sagemaker session are used depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues are seen.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param estimator_parameter: Estimator paramerters for SM job.
    :param inputs: Inputs for fit estimator call
    :param disable_sm_profiler: Flag to disable SM profiler
    :param upload_s3_data_args: Data to be uploded to S3 for training job
    :param job_name: Training job name

    :return: None
    """

    num_retries = 3
    retry_delay = 600
    ecr_image_region = get_ecr_image_region(ecr_image)
    for _ in range(num_retries):
        for test_region in sagemaker_regions:
            sagemaker_session = get_sagemaker_session(test_region)
            # Reupload the image to test region if needed
            tested_ecr_image = get_ecr_image(ecr_image, test_region) if test_region != ecr_image_region else ecr_image
            try:
                pytorch = PyTorch(
                    image_uri=tested_ecr_image,
                    sagemaker_session=sagemaker_session,
                    **estimator_parameter,
                )

                if disable_sm_profiler:
                    if sagemaker_session.boto_region_name in ('cn-north-1', 'cn-northwest-1'):
                        pytorch.disable_profiler = True

                if upload_s3_data_args:
                    training_input = upload_s3_data(pytorch, **upload_s3_data_args)
                    inputs = {'training': training_input}

                pytorch.fit(inputs=inputs, job_name=job_name)
                return pytorch, sagemaker_session

            except sagemaker.exceptions.UnexpectedStatusException as e:
                if "CapacityError" in str(e):
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
