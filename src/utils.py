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
import logging
import sys

import constants

from config import build_config


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


class JobParameters:
    image_types = []
    device_types = []
    py_versions = []
    image_run_test_types = {}

    @staticmethod
    def build_for_all_images():
        JobParameters.image_types = constants.ALL
        JobParameters.device_types = constants.ALL
        JobParameters.py_versions = constants.ALL

    @staticmethod
    def add_image_types(value):
        if JobParameters.image_types != constants.ALL:
            JobParameters.image_types.append(value)

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
    # This import statement has been placed inside this function because it creates a dependency that is unnecessary
    # for local builds and builds outside of Pull Requests.
    from dlc.github_handler import GitHubHandler

    github_handler = GitHubHandler("aws", "deep-learning-containers")
    files = github_handler.get_pr_files_changed(pr_number)
    files = "\n".join(files)
    return files


def update_image_run_test_types(image_build_string, test_type):
    """
    Map the test_types with image_tags or job_type values, we use this mapping in fetch_dlc_images_for_test_jobs
    to append images for each test job
    :param image_build_string: str (image_name or training or inference or all)
    :param test_type: str (all or ec2 or ecs or eks or sagemaker)
    :return:
    """
    if image_build_string in JobParameters.image_run_test_types.keys():
        test = JobParameters.image_run_test_types.get(image_build_string)
        # If image_build_string is already present
        # we will only append the test_type if it doesn't have all tests.
        if constants.ALL not in test and test_type != constants.ALL:
            test.append(test_type)
            JobParameters.image_run_test_types[image_build_string] = test
        # if test_type is "all" then we override existing value with that.
        elif test_type == constants.ALL:
            JobParameters.image_run_test_types[image_build_string] = [test_type]
    # Assigning the test_type to image_build_string for the first time
    else:
        JobParameters.image_run_test_types[image_build_string] = [test_type]


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
        LOGGER.info(f"Building dockerfile: {dockerfile}")
        # Use class static variables to avoid passing, returning the varibles from all functions
        JobParameters.device_types.append(device_type)
        JobParameters.image_types.append(image_type)
        JobParameters.py_versions.append(py_version)
        # create a map for the image_build_string and run_test_types on it
        # this map will be used to update the DLC_IMAGES for pr test jobs
        run_tests_key = f"{image_type}_{device_type}_{py_version}"
        update_image_run_test_types(run_tests_key, constants.ALL)


def parse_modifed_buidspec_yml_info(files, framework, pattern=""):
    """
    trigger a build for all the images related to a framework when there is change in framework/buildspec.yml
    :param files: str
    :param framework: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    for buildspec in rule:
        buildspec_framework = buildspec.split("/")[0]
        if buildspec_framework == framework:
            JobParameters.build_for_all_images()
            update_image_run_test_types(constants.ALL, constants.ALL)


# Rule 3: If any file in the build code changes, build all images
def parse_modifed_root_files_info(files, pattern=""):
    """
    trigger a build for all the images for all the frameworks when there is change in src, test, testspec.yml files
    :param files: str
    :param pattern: str
    :return: None
    """
    rule = re.findall(rf"{pattern}", files)
    if rule:
        JobParameters.build_for_all_images()
        update_image_run_test_types(constants.ALL, constants.ALL)


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
        test_dirs = test_file.split("/")
        test_folder = test_dirs[0]
        if test_folder == "sagemaker_tests":
            framework_changed = test_dirs[1]
            # The below code looks for file changes in /test/sagemaker_tests/(mxnet|pytorch|tensorflow) directory
            if framework_changed == framework:
                job_name = test_dirs[2]
                # The training folder structure for tensorflow is tensorflow1_training(1.x), tensorflow2_training(2.x)
                # so we are stripping the tensorflow1 from the name
                if framework_changed == "tensorflow" and "training" in job_name:
                    job_name = "training"
                if job_name in constants.IMAGE_TYPES:
                    JobParameters.add_image_types(job_name)
                    JobParameters.build_for_all_device_types_py_versions()
                    update_image_run_test_types(job_name, constants.SAGEMAKER_TESTS)
                # If file changed is under /test/sagemaker_tests/(mxnet|pytorch|tensorflow)
                # but not in training/inference dirs
                else:
                    JobParameters.build_for_all_images()
                    update_image_run_test_types(
                        constants.ALL, constants.SAGEMAKER_TESTS
                    )
                    break
            # If file changed is under /test/sagemaker_tests but not in (mxnet|pytorch|tensorflow) dirs
            elif framework_changed not in constants.FRAMEWORKS:
                JobParameters.build_for_all_images()
                update_image_run_test_types(constants.ALL, constants.SAGEMAKER_TESTS)
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
    for test_file in rule:
        test_dirs = test_file.split("/")
        test_folder = test_dirs[0]
        if test_folder == "dlc_tests":
            test_name = test_dirs[1]
            # The below code looks for file changes in /test/dlc_tests/(ecs|eks|ec2) directory
            if test_name in ["ecs", "eks", "ec2"]:
                framework_changed = test_dirs[2]
                if framework_changed == framework:
                    job_name = test_dirs[3]
                    if job_name in constants.IMAGE_TYPES:
                        JobParameters.add_image_types(job_name)
                        JobParameters.build_for_all_device_types_py_versions()
                        update_image_run_test_types(job_name, test_name)
                    # If file changed is under /test/dlc_tests/(ecs|eks|ec2)
                    # but not in (inference|training) dirs
                    else:
                        JobParameters.build_for_all_images()
                        update_image_run_test_types(constants.ALL, test_name)
                        break
                # If file changed is under /test/dlc_tests/(ecs|eks|ec2) dirs init and conftest files
                elif framework_changed not in constants.FRAMEWORKS:
                    JobParameters.build_for_all_images()
                    update_image_run_test_types(constants.ALL, test_name)
                    break
            # If file changed is under /test/dlc_tests/ dir sanity, container_tests dirs
            # and init, conftest files
            else:
                JobParameters.build_for_all_images()
                update_image_run_test_types(constants.ALL, constants.EC2_TESTS)
                update_image_run_test_types(constants.ALL, constants.ECS_TESTS)
                update_image_run_test_types(constants.ALL, constants.EKS_TESTS)
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

    parse_modified_sagemaker_test_files(
        files, framework, pattern="sagemaker_tests\/\S+"
    )

    # The below functions are only run if all JobParameters variables are not set with constants.ALL
    parse_modified_dlc_test_files_info(files, framework, pattern="dlc_tests\/\S+")

    # The below code currently overides the device_types, image_types, py_versions with constants.ALL
    # when there is a change in any the below files
    parse_modifed_buidspec_yml_info(files, framework, pattern="\S+\/buildspec.*yml")

    parse_modifed_root_files_info(files, pattern="src\/\S+")

    parse_modifed_root_files_info(
        files, pattern="(?:test\/(?!(dlc_tests|sagemaker_tests))\S+)"
    )

    parse_modifed_root_files_info(files, pattern="testspec\.yml")

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
    build_context = os.environ.get("BUILD_CONTEXT")

    if build_context == "PR":
        pr_number = os.getenv("CODEBUILD_SOURCE_VERSION")
        LOGGER.info(f"pr number: {pr_number}")
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
                if not build_config.DISABLE_NEW_BUILDS or build_context != "PR":
                    os.environ[env_variable] = "true"


def fetch_dlc_images_for_test_jobs(images):
    """
    use the JobParamters.run_test_types values to pass on image ecr urls to each test type.
    :param images: list
    :return: dictionary
    """
    DLC_IMAGES = {"sagemaker": [], "ecs": [], "eks": [], "ec2": [], "sanity": []}

    for docker_image in images:
        use_preexisting_images = (build_config.DISABLE_NEW_BUILDS and docker_image.build_status == constants.NOT_BUILT)
        if docker_image.build_status == constants.SUCCESS or use_preexisting_images:
            if "mxnet-inference" in docker_image.ecr_url or "gpu" in docker_image.ecr_url:
                continue
            # Run sanity tests on the all images built
            DLC_IMAGES["sanity"].append(docker_image.ecr_url)
            image_job_type = docker_image.info.get("image_type")
            image_device_type = docker_image.info.get("device_type")
            image_python_version = docker_image.info.get("python_version")
            image_tag = f"{image_job_type}_{image_device_type}_{image_python_version}"
            # when image_run_test_types has key all values can be (all , ecs, eks, ec2, sagemaker)
            if constants.ALL in JobParameters.image_run_test_types.keys():
                run_tests = JobParameters.image_run_test_types.get(constants.ALL)
                run_tests = (
                    constants.ALL_TESTS if constants.ALL in run_tests else run_tests
                )
                for test in run_tests:
                    DLC_IMAGES[test].append(docker_image.ecr_url)
            # when key is training or inference values can be  (ecs, eks, ec2, sagemaker)
            if image_job_type in JobParameters.image_run_test_types.keys():
                run_tests = JobParameters.image_run_test_types.get(image_job_type)
                for test in run_tests:
                    DLC_IMAGES[test].append(docker_image.ecr_url)
            # when key is image_tag (training-cpu-py3) values can be (ecs, eks, ec2, sagemaker)
            if image_tag in JobParameters.image_run_test_types.keys():
                run_tests = JobParameters.image_run_test_types.get(image_tag)
                run_tests = (
                    constants.ALL_TESTS if constants.ALL in run_tests else run_tests
                )
                for test in run_tests:
                    DLC_IMAGES[test].append(docker_image.ecr_url)

    for test_type in DLC_IMAGES.keys():
        test_images = DLC_IMAGES[test_type]
        if test_images:
            DLC_IMAGES[test_type] = list(set(test_images))
    return DLC_IMAGES


def write_to_json_file(file_name, content):
    with open(file_name, "w") as fp:
        json.dump(content, fp)


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

    test_images_dict = fetch_dlc_images_for_test_jobs(images)

    # dumping the test_images to dict that can be used in src/start_testbuilds.py
    write_to_json_file(constants.TEST_TYPE_IMAGES_PATH, test_images_dict)

    LOGGER.debug(f"Utils Test Type Images: {test_images_dict}")

    if kwargs:
        for key, value in kwargs.items():
            test_envs.append({"name": key, "value": value, "type": "PLAINTEXT"})

    write_to_json_file(constants.TEST_ENV_PATH, test_envs)


def get_codebuild_project_name():
    return os.getenv("CODEBUILD_BUILD_ID").split(":")[0]
