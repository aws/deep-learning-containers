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


def get_efa_test_instance_type(default: list):
    """
    Get the instance type to be used for EFA tests from the environment, or default to a given value if the type
    isn't specified in the environment.

    :param default: list of instance type to be used for tests
    :return: list of instance types to be parametrized for a test
    """
    configured_instance_type = os.getenv("SM_EFA_TEST_INSTANCE_TYPE")
    if configured_instance_type:
        return [configured_instance_type]
    return default
