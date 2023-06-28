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

import re
import time

import boto3
import botocore.exceptions
import sagemaker

from botocore.config import Config
from sagemaker.session import Session
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_delay

from test.test_utils.ecr import reupload_image_to_test_ecr

# It is possible to have such low capacity on certain instance types that the test is never able to
# run due to ICE errors. In these cases, we are forced to xfail/skip the test, or end up causing
# pipelines to fail forever. We have approval to skip the test when this type of ICE error occurs
# for p4de. Will need approval for each new instance type to be added to this list.
LOW_AVAILABILITY_INSTANCE_TYPES = ["ml.p4de.24xlarge"]


class SMInstanceCapacityError(Exception):
    pass


class SMResourceLimitExceededError(Exception):
    pass


class SMThrottlingError(Exception):
    pass


def get_sagemaker_session(region, default_bucket=None):
    return sagemaker.Session(
        boto_session=boto3.Session(region_name=region), default_bucket=default_bucket
    )


def get_sagemaker_client(region):
    return boto3.Session(region_name=region).client(
        "sagemaker", config=Config(retries={"max_attempts": 10})
    )


def get_sagemaker_runtime_client(region):
    return boto3.Session(region_name=region).client(
        "runtime.sagemaker", config=Config(retries={"max_attempts": 10})
    )


def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.

    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub("[^A-Za-z0-9]+", "", image_uri)


def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


def get_ecr_image_region(ecr_image):
    ecr_registry, _ = ecr_image.split("/")
    region_search = re.search(
        r"(us(-gov)?|af|ap|ca|cn|eu|il|me|sa)-(central|(north|south)?(east|west)?)-\d+",
        ecr_registry,
    )
    return region_search.group()


def get_ecr_image(ecr_image, region):
    """
    It uploads image to the aws region and return image uri
    """
    image_repo_uri, image_tag = ecr_image.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    target_image_repo_name = f"{image_repo_name}"
    regional_ecr_image = reupload_image_to_test_ecr(ecr_image, target_image_repo_name, region)
    return regional_ecr_image


@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (SMInstanceCapacityError, SMResourceLimitExceededError, SMThrottlingError)
    ),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_sm_helper_function(ecr_image, sagemaker_regions, test_function, *test_function_args):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and
    the sagemaker session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues occur.

    E.g
    invoke_sm_helper_function(ecr_image, sagemaker_regions, test_function_to_be_executed,
                                test_function_arg1, test_function_arg2, test_function_arg3)

    That way {@param test_function_to_be_executed} will be sequentially executed in
    {@param sagemaker_regions} with all provided test_function_args

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param test_function: Function to invoke
    :param test_function_args: Helper function args

    :return: None
    """

    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None
    for region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        )
        try:
            test_function(tested_ecr_image, sagemaker_session, *test_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                error = e
                continue
            else:
                raise e
        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e) or "ResourceLimitExceeded" in str(e):
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


@retry(
    reraise=True,
    retry=retry_if_exception_type(
        (SMInstanceCapacityError, SMResourceLimitExceededError, SMThrottlingError)
    ),
    stop=stop_after_delay(20 * 60),
    wait=wait_fixed(60),
)
def invoke_sm_endpoint_helper_function(
    ecr_image,
    sagemaker_regions,
    test_function,
    local_model_paths=None,
    model_helper=None,
    mme_folder_name=None,
    dump_logs_from_cloudwatch=None,
    **test_function_args,
):
    ecr_image_region = get_ecr_image_region(ecr_image)
    error = None
    for region in sagemaker_regions:
        sagemaker_client = get_sagemaker_client(region)
        boto_session = boto3.Session(region_name=region)
        sagemaker_runtime_client = get_sagemaker_runtime_client(region)
        sagemaker_session = Session(boto_session=boto3.Session(region_name=region))
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        )

        # Run model setup if needed
        tested_model_data = None
        if model_helper and local_model_paths:
            if not mme_folder_name:
                tested_model_data = model_helper(
                    region=region, boto_session=boto_session, local_path=local_model_paths[0]
                )
            else:
                tested_model_data = model_helper(
                    region=region,
                    boto_session=boto_session,
                    mme_folder_name=mme_folder_name,
                    path_list=local_model_paths,
                )

        # Run test function
        try:
            return_value = test_function(
                image_uri=tested_ecr_image,
                boto_session=boto_session,
                sagemaker_client=sagemaker_client,
                sagemaker_runtime_client=sagemaker_runtime_client,
                sagemaker_session=sagemaker_session,
                model_data=tested_model_data,
                region=region,
                **test_function_args,
            )
            return return_value
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                error = e
                continue
            else:
                if dump_logs_from_cloudwatch:
                    dump_logs_from_cloudwatch(e, region)
                raise e
        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e) or "ResourceLimitExceeded" in str(e):
                error = e
                continue
            else:
                if dump_logs_from_cloudwatch:
                    dump_logs_from_cloudwatch(e, region)
                raise e
        except Exception as e:
            if dump_logs_from_cloudwatch:
                dump_logs_from_cloudwatch(e, region)
            raise
    if "CapacityError" in str(error):
        raise SMInstanceCapacityError from error
    elif "ResourceLimitExceeded" in str(error):
        raise SMResourceLimitExceededError from error
    else:
        raise SMThrottlingError from error
