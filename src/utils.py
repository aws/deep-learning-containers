"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""

import os
import re

import constants
from github import GitHubHandler


def pr_build_setup(pr_number, framework):
    """
    Identify the PR changeset and set the appropriate environment
    variables

    Parameters:
        pr_number: int

    Returns:
        device_types: [str]
        image_types: [str]
        py_versions: [str]
    """
    github_handler = GitHubHandler("aws", "deep-learning-containers")
    files = github_handler.get_pr_files_changed(pr_number)
    files = "\n".join(files)

    device_types = []
    image_types = []
    py_versions = []

    # Rule 1: Build only those images whose dockerfiles have changed
    rule = re.findall(r"\S+Dockerfile\S+", files)
    for dockerfile in rule:

        dockerfile = dockerfile.split('/')
        image_type = dockerfile[1]
        py_version = dockerfile[4]
        device_type = dockerfile[-1].split(".")

        device_types.append(device_type)
        image_types.append(image_type)
        py_versions.append(py_version)

    # Rule 2: If the buildspec file for a framework changes, build all images for that framework
    rule = re.findall(r"\S+\/buildspec.yml", files)
    for buildspec in rule:
        buildspec_framework, _ = buildspec.split("/")
        if buildspec_framework == framework:
            device_types = constants.ALL
            image_types = constants.ALL
            py_versions = constants.ALL

    # Rule 3: If any file in the build code changes, build all images
    rule = re.findall(r"src\/\S+", files)
    if len(rule) != 0:
        device_types = constants.ALL
        image_types = constants.ALL
        py_versions = constants.ALL

    return device_types, image_types, py_versions


def build_setup(framework, device_types=None, image_types=None, py_versions=None):
    """
    Setup the appropriate environment variables depending on whether this is a PR build
    or a dev build

    Parameters:
        framework: str
        device_types: [str]
        image_types: [str]
        py_versions: [str]

    Returns:
        None
    """

    # Set necessary environment variables
    to_build = {
        "device_types": constants.DEVICE_TYPES,
        "image_types": constants.IMAGE_TYPES,
        "py_versions": constants.PYTHON_VERSIONS,
    }

    if os.environ.get("BUILD_CONTEXT") == "PR":
        pr_number = "pr/16" #os.getenv("CODEBUILD_SOURCE_VERSION")
        if pr_number is not None:
            pr_number = int(pr_number.split("/")[-1])
        device_types, image_types, py_versions = pr_build_setup(
            pr_number, framework
        )

    if device_types != constants.ALL:
        to_build["device_types"] = constants.DEVICE_TYPES.intersection(
            set(device_types)
        )
    if image_types != constants.ALL:
        to_build["image_types"] = constants.IMAGE_TYPES.intersection(set(image_types))
    if py_versions != constants.ALL:
        to_build["py_versions"] = constants.PYTHON_VERSIONS.intersection(
            set(py_versions)
        )

    for device_type in to_build["device_types"]:
        for image_type in to_build["image_types"]:
            for py_version in to_build["py_versions"]:
                env_variable = f"{framework.upper()}_{device_type.upper()}_{image_type.upper()}_{py_version.upper()}"
                os.environ[env_variable] = "true"
