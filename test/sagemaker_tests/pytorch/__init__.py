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

import botocore.exceptions
import sagemaker
import sagemaker.exceptions
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_fixed

NO_P4_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
    "cn-northwest-1",
    "il-central-1",
]

NO_G5_REGIONS = [
    "us-west-1",
    "ca-west-1",
    "mx-cental-1",
    "af-south-1",
    "ap-east-1",
    "ap-south-2",
    "ap-southeast-5",
    "ap-southeast-4",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-7",
    "eu-south-1",
    "eu-west-3",
    "eu-south-2",
    "eu-central-2",
    "me-south-1",
]

P5_AVAIL_REGIONS = [
    "us-east-1",
    "us-west-2",
]


class SMInstanceCapacityError(Exception):
    pass


class SMResourceLimitExceededError(Exception):
    pass


class SMThrottlingError(Exception):
    pass


@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (SMInstanceCapacityError, SMResourceLimitExceededError, SMThrottlingError)
    ),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_pytorch_helper_function(
    ecr_image, sagemaker_regions, helper_function, helper_function_args
):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and
    the sagemaker session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues occur.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param helper_function: Function to invoke
    :param helper_function_args: Helper function args

    :return: None
    """
    from .. import get_ecr_image, get_ecr_image_region, get_sagemaker_session

    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None
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
                error = e
                continue
            else:
                raise e
        except botocore.exceptions.ClientError as e:
            if any(
                exception_type in str(e)
                for exception_type in ["ThrottlingException", "ResourceLimitExceeded"]
            ):
                error = e
                continue
            else:
                raise e
    if "CapacityError" in str(error):
        raise SMInstanceCapacityError from error
    elif "ResourceLimitExceeded" in str(error):
        raise SMResourceLimitExceededError from error
    else:
        raise SMThrottlingError from error
