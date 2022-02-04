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

import boto3
import sagemaker

from test.test_utils.ecr import reupload_image_to_test_ecr, get_ecr_image_labels_from_uri


def get_sagemaker_session(region, default_bucket=None):
    return sagemaker.Session(boto_session=boto3.Session(region_name=region), default_bucket=default_bucket)


def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.

    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub('[^A-Za-z0-9]+', '', image_uri)


def get_ecr_image_region(ecr_image):
    ecr_registry, _ = ecr_image.split("/")
    region_search = re.search(r"(us(-gov)?|ap|ca|cn|eu|sa|me|af)-(central|(north|south)?(east|west)?)-\d+", ecr_registry)
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


def invoke_sm_helper_function(ecr_image, sagemaker_regions, test_function, *test_function_args):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and the sagemaker
    session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues are seen.

    E.g
    invoke_sm_helper_function(ecr_image, sagemaker_regions, test_function_to_be_executed,
                                test_function_arg1, test_function_arg2, test_function_arg3)

    That way {@param test_function_to_be_executed} will be sequentially executed in {@param sagemaker_regions}
    with all provided test_function_args

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param test_function: Function to invoke
    :param test_function_args: Helper function args

    :return: None
    """

    ecr_image_region = get_ecr_image_region(ecr_image)
    for region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        try:
            test_function(tested_ecr_image, sagemaker_session, *test_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                continue
            else:
                raise e


def is_image_smddp_compatible(image_uri):
    """
    Imperfect label-check to skip tests when images do not have SageMaker Distributed Data Parallelism installed
    :param image_uri:
    :return:
    """
    image_labels = get_ecr_image_labels_from_uri(image_uri)
    return image_labels.get("smddp_installed", "true").lower() == "true"


def is_image_smdmp_compatible(image_uri):
    """
    Imperfect label-check to skip tests when images do not have SageMaker Distributed Model Parallelism installed
    :param image_uri:
    :return:
    """
    image_labels = get_ecr_image_labels_from_uri(image_uri)
    return image_labels.get("smdmp_installed", "true").lower() == "true"


def is_image_smdebug_compatible(image_uri):
    """
    Imperfect label-check to skip tests when images do not have SageMaker Debugger
    :param image_uri:
    :return:
    """
    image_labels = get_ecr_image_labels_from_uri(image_uri)
    return image_labels.get("smdebug_installed", "true").lower() == "true"
