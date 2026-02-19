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
    # Returns str, not Version(). PEP 440 normalizes 24.04 -> 24.4, losing the
    # leading zero which is significant in Ubuntu CalVer
    return match.group(1)


@pytest.fixture(scope="session")
def framework_name(image_uri):
    """Parse framework name from image URI (e.g., pytorch, vllm, sglang, base)."""
    if not image_uri:
        raise ValueError("No image URI provided")
    for name in FRAMEWORK_MODULE_MAP:
        if name in image_uri:
            return name
    raise ValueError(f"No known framework found in image URI: {image_uri}")


@pytest.fixture(scope="session")
def framework_module(framework_name):
    """Map framework to Python module name (e.g., pytorch -> torch)."""
    module = FRAMEWORK_MODULE_MAP[framework_name]
    if module is None:
        pytest.skip(f"Framework '{framework_name}' has no Python module")
    return module


@pytest.fixture(scope="session")
def framework_version(framework_module, image_uri):
    """Parse framework version from image URI tag (first version-like segment)."""
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
