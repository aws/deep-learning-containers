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
"""Sanity tests for DLC pre-release validation"""

import json
import logging
import os
import re
import subprocess
from pprint import pformat

import pytest
from packaging.version import Version

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ALLOWLIST_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "allowlists")


def _load_allowlist(name, framework=None, image_tag=None):
    """Load and merge 3-level allowlist: global -> framework -> image-specific."""
    entries = []
    paths = [os.path.join(ALLOWLIST_DIR, f"{name}.json")]
    if framework:
        paths.append(os.path.join(ALLOWLIST_DIR, framework, f"{name}.json"))
    if framework and image_tag:
        paths.append(os.path.join(ALLOWLIST_DIR, framework, f"{image_tag}.json"))
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                entries.extend(json.load(f))
    return entries


def test_python_version(python_version):
    """Verify installed Python version matches image tag."""
    result = subprocess.run(
        ["python3", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    installed = result.stdout.strip()
    LOGGER.info("Installed Python: %s", installed)
    assert str(python_version) in installed, f"Expected Python {python_version}, got: {installed}"


def test_ubuntu_version(ubuntu_version):
    """Verify Ubuntu version matches image tag."""
    with open("/etc/os-release") as f:
        os_release = f.read()
    assert "Ubuntu" in os_release, "OS is not Ubuntu"
    assert str(ubuntu_version) in os_release, (
        f"Expected Ubuntu {ubuntu_version} not found in /etc/os-release"
    )


def test_framework_version(framework_module, framework_version):
    """Verify installed framework version matches image tag."""
    result = subprocess.run(
        ["python3", "-c", f"import {framework_module}; print({framework_module}.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    installed = Version(result.stdout.strip().split("+")[0])
    LOGGER.info("Installed %s version: %s", framework_module, installed)
    assert str(installed).startswith(str(framework_version)), (
        f"Expected {framework_module} {framework_version}, got: {installed}"
    )


def test_cuda_version(cuda_version):
    """Verify installed CUDA version matches image tag."""
    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail("nvcc not found — expected CUDA image but nvcc is missing")
    cuda_output = result.stdout.replace(".", "")
    expected = str(cuda_version).replace(".", "")
    assert expected in cuda_output, (
        f"Expected CUDA {cuda_version} not found in nvcc output:\n{result.stdout}"
    )


def test_pip_check(framework_name, image_uri):
    """Verify no broken pip dependencies, with allowlist support."""
    image_tag = image_uri.split(":")[-1] if image_uri and ":" in image_uri else None
    allowlist = _load_allowlist("pip_check", framework=framework_name, image_tag=image_tag)
    allowed_patterns = [entry["pattern"] for entry in allowlist]

    result = subprocess.run(
        ["pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    failures = []
    for line in result.stdout.strip().splitlines():
        if not any(re.search(p, line) for p in allowed_patterns):
            failures.append(line)

    assert not failures, f"pip check found broken dependencies:\n{pformat(failures)}"


def test_oss_compliance():
    """Verify license attribution files exist in the container."""
    result = subprocess.run(
        ["/usr/local/bin/testOSSCompliance", "/root"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"OSS compliance check failed:\n{result.stderr}"
