# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest
from sagemaker.remote_function import remote

from .timeout import timeout

from test.test_utils import get_python_version_from_image_uri
from ..... import invoke_sm_helper_function


RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
DEPENDENCIES_PATH = os.path.join(RESOURCE_PATH, "remote_function", "requirements.txt")
DEFAULT_TIMEOUT = 20


@pytest.mark.model("N/A")
@pytest.mark.integration("remote_function")
@pytest.mark.skip_py2_containers
def test_remote_function_divide(ecr_image, sagemaker_regions, instance_type):
    """Test if the image works with Sagemaker remote_function"""

    python_version = get_python_version_from_image_uri(ecr_image).replace("py", "")
    python_version = int(python_version)

    if python_version < 37:
        pytest.skip("remote_function test is for Python>= 3.7")

    instance_type = instance_type or "ml.m5.xlarge"

    invoke_sm_helper_function(
        ecr_image, sagemaker_regions, _test_remote_function_training, instance_type
    )


def _test_remote_function_training(ecr_image, sagemaker_session, instance_type):
    @remote(
        role="SageMakerRole",
        image_uri=ecr_image,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        dependencies=DEPENDENCIES_PATH,
    )
    def dlc_remote_function_test_divide(x, y):
        return x / y

    with timeout(minutes=DEFAULT_TIMEOUT):
        assert dlc_remote_function_test_divide(10, 2) == 5
