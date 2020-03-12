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


class JobParameters:
    image_types = []
    device_types = []
    py_versions = []

    @staticmethod
    def build_for_all_images():
        JobParameters.image_types = constants.ALL
        JobParameters.device_types = constants.ALL
        JobParameters.py_versions = constants.ALL

    @staticmethod
    def build_for_all_device_types_py_versions():
        JobParameters.device_types = constants.ALL
        JobParameters.py_versions = constants.ALL

    @staticmethod
    def do_build_all_images():
        return (
            JobParameters.device_types == constants.ALL
            and JobParameters.image_types == constants.ALL
            and JobParameters.py_versions == constants.ALL
        )


def get_pr_modified_files(pr_number):
    """
    Fetch all the files modified for a git pull request and return them as a string
    :param pr_number: int
    :return: str with all the modified files
    """
    github_handler = GitHubHandler("aws", "deep-learning-containers")
    files = github_handler.get_pr_files_changed(pr_number)
    files = "\n".join(files)
    return files


def parse_modified_docker_files_info(files, framework, pattern=""):
    """
    Parse all the files in PR to find docker file related changes for any framework
    triggers an image build matching the image_type(training/testing), device_type(cpu_gpu)
    and python version(py2 and py3) of the changed docker files
    :param files: str
    :param framework: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
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
        # Use class static variables to avoid passing, returning the varibles from all functions
        JobParameters.device_types.append(device_type)
        JobParameters.image_types.append(image_type)
        JobParameters.py_versions.append(py_version)


def parse_modifed_buidspec_yml_info(files, framework, pattern=""):
    """
    trigger a build for all the images related to a framework when there is change in framework/buildspec.yml
    :param files: str
    :param framework: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    if not JobParameters.do_build_all_images():
        for buildspec in rule:
            buildspec_framework = buildspec.split("/")[0]
            if buildspec_framework == framework:
                JobParameters.build_for_all_images()


# Rule 3: If any file in the build code changes, build all images
def parse_modifed_root_files_info(files, pattern=""):
    """
    trigger a build for all the images for all the frameworks when there is change in src, test, testspec.yml files
    :param files: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    if not JobParameters.do_build_all_images():
        if rule:
            JobParameters.build_for_all_images()


def parse_modified_sagemaker_test_files(files, framework, pattern=""):
    """
    Parse all the files in PR to find sagemaker tests related changes for any framework
    to trigger an image build matching the image_type(training/testing) for all the device_types(cpu,gpu)
    and python_versions(py2,py3)
    :param files: str
    :param framework: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    for test_file in rule:
        test_folder = test_file.split("/")[1]
        if test_folder == "sagemaker_tests":
            framework_changed = test_file.split("/")[2]
            # The below code looks for file changes in /test/sagemaker_tests/(mxnet|pytorch|tensorflow) directory
            if framework_changed == framework:
                job_name = test_file.split("/")[3]
                # The training folder structure for tensorflow is tensorflow1_training(1.x), tensorflow2_training(2.x)
                # so we are stripping the tensrflow1 from the name
                if framework_changed == "tensorflow" and "training" in job_name:
                    job_name = "training"
                if job_name in constants.IMAGE_TYPES:
                    JobParameters.image_types.append(job_name)
                    JobParameters.build_for_all_device_types_py_versions()
                else:
                    JobParameters.build_for_all_images()
                    break
            # If file changed is under /test/sagemaker_tests but not in (mxnet|pytorch|tensorflow) dirs
            elif framework_changed not in constants.FRAMEWORKS:
                JobParameters.build_for_all_images()
                break


def parse_modified_dlc_test_files_info(files, framework, pattern=""):
    """
    Parse all the files in PR to find ecs/eks/ec2 tests related changes for any framework
    to trigger an image build matching the image_type(training/testing) for all the device_types(cpu,gpu)
    and python_versions(py2,py3)
    :param files:
    :param framework:
    :param pattern:
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    # JobParameters variables are not set with constants.ALL
    if not JobParameters.do_build_all_images():
        for test_file in rule:
            test_folder = test_file.split("/")[1]
            if test_folder == "dlc_tests":
                test_name = test_file.split("/")[2]
                # The below code looks for file changes in /test/dlc_tests/(ecs|eks|ec2) directory
                if test_name in ["ecs", "eks", "ec2"]:
                    framework_changed = test_file.split("/")[3]
                    if framework_changed == framework:
                        job_name = test_file.split("/")[4]
                        if job_name in constants.IMAGE_TYPES:
                            JobParameters.image_types.append(job_name)
                            JobParameters.build_for_all_device_types_py_versions()
                        # If file changed is under /test/sagemaker_tests/(ecs|eks|ec2)
                        # but not in (mxnet|pytorch|tensorflow) dirs
                        else:
                            JobParameters.build_for_all_images()
                            break
                    # If file changed is under /test/dlc_tests but not in (ecs|eks|ec2) dirs
                    elif framework_changed not in constants.FRAMEWORKS:
                        JobParameters.build_for_all_images()
                        break


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

    # This below code currently appends the values to device_types, image_types, py_versions for files changed
    # if there are no changes in the files then functions return same lists
    parse_modified_docker_files_info(files, framework, pattern="\S+Dockerfile\S+")

    # TODO we have re enable this logic to parse test files once test migration is done
    # parse_modified_sagemaker_test_files(
    #     files, framework, pattern="\S+sagemaker_tests\/\S+"
    # )

    # The below functions are only run if all JobParameters variables are not set with constants.ALL
    # TODO we have re enable this logic to parse test files once test migration is done
    # parse_modified_dlc_test_files_info(files, framework, pattern="\S+dlc_tests\/\S+")

    # The below code currently overides the device_types, image_types, py_versions with constants.ALL
    # when there is a change in any the below files
    parse_modifed_buidspec_yml_info(files, framework, pattern="\S+\/buildspec.yml")

    parse_modifed_root_files_info(files, pattern="src\/\S+")

    # TODO we have re enable this logic after test migration is done
    # parse_modifed_root_files_info(
    #     files, pattern="(?:test\/(?!(dlc_tests|sagemaker_tests))\S+)"
    # )

    # TODO we have re enable this logic after test migration is done
    # parse_modifed_root_files_info(files, pattern="testspec\.yml")

    return (
        JobParameters.device_types,
        JobParameters.image_types,
        JobParameters.py_versions,
    )


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

    for docker_image in images:
        # TODO we have change this logic after to append urls only for new builds after test migration is done
        ecr_urls.append(docker_image.ecr_url)

    images_arg = " ".join(ecr_urls)
    test_envs.append({"name": images_env, "value": images_arg, "type": "PLAINTEXT"})

    if kwargs:
        for key, value in kwargs.items():
            test_envs.append({"name": key, "value": value, "type": "PLAINTEXT"})

    with open(constants.TEST_ENV, "w") as ef:
        json.dump(test_envs, ef)


def get_codebuild_project_name():
    return os.getenv("CODEBUILD_BUILD_ID").split(":")[0]
