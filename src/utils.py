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

def get_pr_modified_files(pr_number):
    """
    Fetches all the files modified for a git pull request and return them as a string
    :param pr_number:
    :return: str with all the modified files
    """
    github_handler = GitHubHandler("aws", "deep-learning-containers")
    files = github_handler.get_pr_files_changed(pr_number)
    files = "\n".join(files)
    return files

def get_modified_docker_files_info(files, framework, device_types=None, image_types=None, py_versions=None):
    if py_versions is None:
        py_versions = []
    if image_types is None:
        image_types = []
    if device_types is None:
        device_types = []
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
    return  device_types, image_types, py_versions

def get_modifed_buidspec_yml_info(files, framework, device_types=None, image_types=None, py_versions=None):
    if py_versions is None:
        py_versions = []
    if image_types is None:
        image_types = []
    if device_types is None:
        device_types = []
    rule = re.findall(r"\S+\/buildspec.yml", files)
    for buildspec in rule:
        buildspec_framework = buildspec.split("/")[0]
        if buildspec_framework == framework:
            device_types = constants.ALL
            image_types = constants.ALL
            py_versions = constants.ALL
    return device_types, image_types, py_versions

# Rule 3: If any file in the build code changes, build all images
def get_modifed_src_files_info(files, device_types=None, image_types=None, py_versions=None):
    if py_versions is None:
        py_versions = []
    if image_types is None:
        image_types = []
    if device_types is None:
        device_types = []
    rule = re.findall(r"src\/\S+", files)
    if rule:
        device_types = constants.ALL
        image_types = constants.ALL
        py_versions = constants.ALL
    return device_types, image_types, py_versions

def get_modified_test_files_info(files, framework, device_types=None, image_types=None, py_versions=None,
                                 run_test_types=None):
    # Rule 1: run  only the tests where the test_files are changed
    if run_test_types is None:
        run_test_types = []
    if py_versions is None:
        py_versions = []
    if image_types is None:
        image_types = []
    if device_types is None:
        device_types = []
    rule = re.findall(r"[\r\n]+test\S+", files)
    for test_file in rule:
        test_folder = test_file.split("/")[1]

        if test_folder == "sagemaker_tests":
            framework_changed = test_file.split("/")[2]
            job_name = test_file.split("/")[3]
            if framework_changed != framework:
                continue
            run_test_types.append(constants.SAGEMAKER_TESTS)
            image_types.append(job_name)
            device_types = constants.ALL
            py_versions = constants.ALL

        elif test_folder == "dlc_tests":
            framework_changed = test_file.split("/")[3]
            test_name = test_file.split("/")[2]
            job_name = test_file.split("/")[4]
            if framework_changed != framework:
                continue
            image_types.append(job_name)
            if test_name == "ecs":
                run_test_types.append(constants.ECS_TESTS)
            elif test_name == "eks":
                run_test_types.append(constants.EKS_TESTS)
             # Assumes that changes are made in dlc_tests but not under ecs or eks folders so we run both the tests
            else:
                run_test_types.extend([constants.EKS_TESTS, constants.ECS_TESTS])
            device_types = constants.ALL
            py_versions = constants.ALL


        else:
            image_types = [constants.ALL]
            run_test_types = [constants.ALL]
            device_types = constants.ALL
            py_versions = constants.ALL

    return device_types, image_types, py_versions, run_test_types


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
    files = get_pr_modified_files(pr_number)

    device_types = []
    image_types = []
    py_versions = []

    # This below code currently appends the values to device_types, image_types, py_versions for files changed
    # if there are no changes in the files then functions return same lists
    # TODO: use a class to define these lists and use getter setter methods
    device_types, image_types, py_versions = get_modified_docker_files_info(files, framework, device_types,
                                                                            image_types, py_versions)

    device_types, image_types, py_versions, run_test_types = get_modified_test_files_info(files, framework, device_types,
                                                                            image_types, py_versions)

    # This below code currently overides the device_types, image_types, py_versions with constants.ALL
    device_types, image_types, py_versions = get_modifed_buidspec_yml_info(files, framework, device_types,
                                                                           image_types, py_versions)

    device_types, image_types, py_versions = get_modifed_src_files_info(files,device_types, image_types,
                                                                        py_versions)

    return device_types, image_types, py_versions


def pr_test_setup(pr_number):
    """
       Identify the PR changeset and set the appropriate environment
       variables

       Parameters:
           pr_number: int

       Returns:
           framwork_changed: [str]
           image_types: [str] possible values ["training","inference"]
           py_versions: [str] possible values ["sagemaker","ecs","eks"]
       """
    github_handler = GitHubHandler("aws", "deep-learning-containers")
    files = github_handler.get_pr_files_changed(pr_number)
    files = "\n".join(files)
    framework = constants.JOB_FRAMEWORK
    framework_version = constants.JOB_FRAMEWORK_VERSION

    device_types, image_types, py_versions, run_test_types = get_modified_test_files_info(files, framework)

    rule1 = re.findall(r"testspec.yml", files)
    if len(rule1) > 0:
        image_types = [constants.ALL]
        run_test_types = [constants.ALL]

    image_types = list(set(image_types))
    run_test_types = list(set(run_test_types))

    return framework, image_types, run_test_types


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
        else:
            raise ValueError ("Empty code commit value found for PR build")
        framework, image_types, run_test_types = pr_test_setup(pr_number)
        print(f"{run_test_types} should be triggered for {framework} and {image_types} ")

    for docker_image in images:
        docker_image_type = docker_image.repository.split("-")[-1]
        if (
            docker_image.build_status == constants.SUCCESS
            or constants.ALL in image_types
        ):
            ecr_urls.append(docker_image.ecr_url)
            run_test_types = [constants.ALL]
        elif (
            framework
            and len(image_types) > 0
            and framework in docker_image.repository
            and docker_image_type in image_types
        ):
            ecr_urls.append(docker_image.ecr_url)
        else:
            print(
                f"skipping tests for {docker_image.ecr_url} as there are no build and test changes"
            )

    images_arg = " ".join(ecr_urls)
    test_envs.append({"name": images_env, "value": images_arg, "type": "PLAINTEXT"})

    if len(run_test_types) > 0:
        with open(constants.RUN_TESTS_ENV, "w") as f:
            for test in run_test_types:
                f.write(f"{test}\n")

    if kwargs:
        for key, value in kwargs.items():
            test_envs.append({"name": key, "value": value, "type": "PLAINTEXT"})

    with open(constants.TEST_ENV, "w") as ef:
        json.dump(test_envs, ef)


def get_codebuild_project_name():
    return os.getenv("CODEBUILD_BUILD_ID").split(":")[0]
