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
"""Common pytest fixtures for all tests under module test/"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--image-uri", action="store", help="Image URI to be tested")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")
