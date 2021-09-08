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
import sagemaker


def invoke_pytorch_helper_function(ecr_image, sagemaker_regions, helper_function, helper_function_args):
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
    from sagemaker_tests import get_ecr_image_region, get_sagemaker_session, get_ecr_image

    ecr_image_region = get_ecr_image_region(ecr_image)
    for region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        try:
            helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                continue
            else:
                raise e
