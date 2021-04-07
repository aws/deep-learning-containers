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

import os
import sys
import pytest
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


@pytest.fixture(autouse=True)
def log_current_test():
    """
    Log the name of the test currently being executed by pytest
    """
    test_name = f"{os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]}"
    LOGGER.info(f"============================= Executing test :: {test_name} :: =============================")
