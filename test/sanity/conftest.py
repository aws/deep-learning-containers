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
"""Pytest fixtures for sanity tests"""

import re

import pytest
from packaging.version import Version
from test_utils.constants import FRAMEWORK_MODULE_MAP


@pytest.fixture(scope="session")
def python_version(image_uri):
    """Parse python version from image URI tag (e.g., -py310- -> 3.10)."""
    match = re.search(r"-py(\d)(\d+)-", image_uri or "")
    if not match:
        raise ValueError(f"No python version found in image URI: {image_uri}")
    return Version(f"{match.group(1)}.{match.group(2)}")


@pytest.fixture(scope="session")
def ubuntu_version(image_uri):
    """Parse ubuntu version from image URI tag (e.g., -ubuntu22.04- -> 22.04)."""
    match = re.search(r"ubuntu(\d+\.\d+)", image_uri or "")
    if not match:
        raise ValueError(f"No ubuntu version found in image URI: {image_uri}")
    return Version(match.group(1))


@pytest.fixture(scope="session")
def framework_name(image_uri):
    """Parse framework name from image URI and map to Python module name."""
    if not image_uri:
        pytest.skip("No image URI provided")
    for framework, module in FRAMEWORK_MODULE_MAP.items():
        if framework in image_uri:
            return module
    pytest.skip(f"No known framework found in image URI: {image_uri}")


@pytest.fixture(scope="session")
def framework_version(image_uri):
    """Parse framework version from image URI tag (first version-like segment)."""
    if not image_uri:
        pytest.skip("No image URI provided")
    tag = image_uri.split(":")[-1]
    match = re.match(r"(\d+\.\d+\.\d+)", tag)
    if not match:
        pytest.skip("No framework version found in image URI tag")
    return Version(match.group(1))


@pytest.fixture(scope="session")
def cuda_version(image_uri):
    """Parse CUDA version from image URI tag (e.g., -cu124- -> 12.4)."""
    match = re.search(r"-cu(\d+)-", image_uri or "")
    if not match:
        pytest.skip("No CUDA version found in image URI")
    raw = match.group(1)
    return Version(f"{raw[:-1]}.{raw[-1]}")
