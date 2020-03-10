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
import json

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

        dockerfile = dockerfile.split("/")
        framework_change = dockerfile[0]

        # If the modified dockerfile belongs to a different
        # framework, do nothing
        if framework_change != framework:
            continue

        image_type = dockerfile[1]
        py_version = dockerfile[4]
        device_type = dockerfile[-1].split(".")[-1]
        device_types.append(device_type)
        image_types.append(image_type)
        py_versions.append(py_version)

    # Rule 2: If the buildspec file for a framework changes, build all images for that framework
    rule = re.findall(r"\S+\/buildspec.yml", files)
    for buildspec in rule:
        buildspec_framework = buildspec.split("/")[0]
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


def pr_test_setup(pr_number):
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
    framework = os.environ.get("FRAMEWORK")
    framework_version = os.environ.get("VERSION")
    framework_changed = ""
    image_types = []
    run_test_types = []

    rule = re.findall(r"\S+[\r\n]+test\S+", files)
    for test_file in rule:
        test_folder = test_file.split("/")[1]

        if test_folder == "sagemaker_tests":
            framework_changed = test_file.split("/")[2]
            job_name = test_file.split("/")[3]
            if framework_changed != framework:
                continue
            run_test_types.append(constants.SAGEMAKER_TESTS)
            image_types.append(job_name)

        elif test_folder == "dlc_tests":
            framework_changed = test_file.split("/")[3]
            test_name = test_file.split("/")[2]
            job_name = test_file.split("/")[4]
            if framework_changed != framework:
                continue
            run_test_types.append(
                constants.ECS_TESTS
            ) if test_name == "ecs" else run_test_types.append(constants.EKS_TESTS)
            image_types.append(job_name)

        else:
            image_types = [constants.ALL]
            run_test_types = [constants.ALL]

    image_types = list(set(image_types))
    run_test_types = list(set(run_test_types))

    return framework_changed, image_types, run_test_types


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
        pr_number = os.getenv("CODEBUILD_SOURCE_VERSION")
        if pr_number is not None:
            pr_number = int(pr_number.split("/")[-1])
        device_types, image_types, py_versions = pr_build_setup(pr_number, framework)

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


def set_test_env(images, images_env="DLC_IMAGES", **kwargs):
    """
    Util function to write a file to be consumed by test env with necessary environment variables

    ENV variables set by os do not persist, as a new shell is instantiated for post_build steps

    :param images: List of image objects
    :param images_env: Name for the images environment variable
    :param env_file: File to write environment variables to
    :param kwargs: other environment variables to set
    """
    test_envs = []
    ecr_urls = []

    if os.environ.get("BUILD_CONTEXT") == "PR":
        pr_number = os.getenv("CODEBUILD_SOURCE_VERSION")
        if pr_number is not None:
            pr_number = int(pr_number.split("/")[-1])
        framework, image_types, run_test_types = pr_test_setup(pr_number)
        print("inside setup_testenv", framework, image_types, run_test_types)

    for docker_image in images:
        docker_image_type = docker_image.repository.split("-")[-1]
        if docker_image.build_status == constants.SUCCESS or constants.ALL in image_types:
            ecr_urls.append(docker_image.ecr_url)
            run_test_types = [constants.ALL]
        elif framework in docker_image.repository and docker_image_type in image_types:
            ecr_urls.append(docker_image.ecr_url)
        else:
            print(
                f"skipping tests for {docker_image.ecr_url} as there are no build and test changes"
            )

    images_arg = " ".join(ecr_urls)
    test_envs.append({"name": images_env, "value": images_arg, "type": "PLAINTEXT"})

    if len(run_test_types) > 0:
        test_types_arg = " ".join(run_test_types)
        test_envs.append(
            {"name": "RUN_TESTS", "value": test_types_arg, "type": "PLAINTEXT"}
        )

    if kwargs:
        for key, value in kwargs.items():
            test_envs.append({"name": key, "value": value, "type": "PLAINTEXT"})

    with open(constants.TEST_ENV, "w") as ef:
        json.dump(test_envs, ef)


def get_codebuild_project_name():
    return os.getenv("CODEBUILD_BUILD_ID").split(":")[0]
