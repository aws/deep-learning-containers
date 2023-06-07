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
import json
import logging
import os
import sys

import boto3

import constants
import config


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def run_test_job(commit, codebuild_project, images_str=""):
    test_env_file = constants.TEST_ENV_PATH
    if not os.path.exists(test_env_file):
        raise FileNotFoundError(
            f"{test_env_file} not found. This is required to set test environment variables"
            f" for test jobs. Failing the build."
        )

    with open(test_env_file) as test_env_file:
        env_overrides = json.load(test_env_file)

    # For SM tests, if EFA_DEDICATED is True, test job will only launch SM Remote EFA tests,
    # or else will only launch standard/rc tests.
    # For EC2 tests, if EFA_DEDICATED is True, test job will launch both EFA and non-EFA tests,
    # or else will only launch non-EFA tests.
    is_test_efa_dedicated = (
        config.are_sm_efa_tests_enabled() and "sagemaker" in codebuild_project
    ) or (config.is_ec2_efa_test_enabled() and "ec2" in codebuild_project)

    pr_num = os.getenv("PR_NUMBER")
    LOGGER.debug(f"pr_num {pr_num}")
    env_overrides.extend(
        [
            {"name": "DLC_IMAGES", "value": images_str, "type": "PLAINTEXT"},
            {"name": "PR_NUMBER", "value": pr_num, "type": "PLAINTEXT"},
            # NIGHTLY_PR_TEST_MODE is passed as an env variable here because it is more convenient to set this in
            # dlc_developer_config, and imports during test execution are less complicated when there are fewer
            # cross-references between test and src code.
            {
                "name": "NIGHTLY_PR_TEST_MODE",
                "value": str(config.is_nightly_pr_test_mode_enabled()),
                "type": "PLAINTEXT",
            },
            # USE_SCHEDULER is passed as an env variable here because it is more convenient to set this in
            # dlc_developer_config, compared to having another config file under dlc/tests/.
            {
                "name": "USE_SCHEDULER",
                "value": str(config.is_scheduler_enabled()),
                "type": "PLAINTEXT",
            },
            # If EFA_DEDICATED is True, only launch SM Remote EFA tests, else only launch standard/rc tests
            {
                "name": "EFA_DEDICATED",
                "value": str(is_test_efa_dedicated),
                "type": "PLAINTEXT",
            },
            # SM_EFA_TEST_INSTANCE_TYPE is passed to SM test job to pick a matching instance type as defined by user
            {
                "name": "SM_EFA_TEST_INSTANCE_TYPE",
                "value": config.get_sagemaker_remote_efa_instance_type(),
                "type": "PLAINTEXT",
            },
        ]
    )
    LOGGER.debug(f"env_overrides dict: {env_overrides}")

    client = boto3.client("codebuild")
    return client.start_build(
        projectName=codebuild_project,
        environmentVariablesOverride=env_overrides,
        sourceVersion=commit,
    )


def is_test_job_enabled(test_type):
    """
    Check to see if a test job is enabled
    See if we should run the tests based on test types and config options.
    """
    if test_type == constants.SAGEMAKER_TESTS and config.is_sm_remote_test_enabled():
        return True
    if test_type == constants.EC2_TESTS and config.is_ec2_test_enabled():
        return True

    # We have no ECS/EKS/SANITY benchmark tests
    if not config.is_benchmark_mode_enabled():
        if test_type == constants.ECS_TESTS and config.is_ecs_test_enabled():
            return True
        if test_type == constants.EKS_TESTS and config.is_eks_test_enabled():
            return True
        if test_type == constants.SANITY_TESTS and config.is_sanity_test_enabled():
            return True

    return False


def is_test_job_implemented_for_framework(images_str, test_type):
    """
    Check to see if a test job is implemnted and supposed to be executed for this particular set of images
    """
    is_trcomp_image = False
    is_huggingface_trcomp_image = False
    is_huggingface_image = False
    if "huggingface" in images_str:
        if "trcomp" in images_str:
            is_huggingface_trcomp_image = True
        else:
            is_huggingface_image = True
    elif "trcomp" in images_str:
        is_trcomp_image = True

    is_autogluon_image = "autogluon" in images_str

    if (is_huggingface_image or is_autogluon_image) and test_type in [
        constants.EC2_TESTS,
        constants.ECS_TESTS,
        constants.EKS_TESTS,
    ]:
        LOGGER.debug(f"Skipping {test_type} test")
        return False
        # SM Training Compiler has EC2 tests implemented so don't skip
    if is_huggingface_trcomp_image and (
        test_type
        in [
            constants.ECS_TESTS,
            constants.EKS_TESTS,
        ]
        or config.is_benchmark_mode_enabled()
    ):
        LOGGER.debug(f"Skipping {test_type} tests for huggingface trcomp containers")
        return False

    if is_trcomp_image and (
        test_type
        in [
            constants.EKS_TESTS,
        ]
        or config.is_benchmark_mode_enabled()
    ):
        LOGGER.debug(f"Skipping {test_type} tests for trcomp containers")
        return False
    return True


def main():
    build_context = os.getenv("BUILD_CONTEXT")
    if build_context != "PR":
        LOGGER.info(f"Not triggering test jobs from boto3, as BUILD_CONTEXT is {build_context}")
        return

    # load the images for all test_types to pass on to code build jobs
    with open(constants.TEST_TYPE_IMAGES_PATH) as json_file:
        test_images = json.load(json_file)

    # Run necessary PR test jobs
    commit = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")

    for test_type, images in test_images.items():
        # only run the code build test jobs when the images are present
        LOGGER.debug(f"test_type : {test_type}")
        LOGGER.debug(f"images: {images}")
        if images:
            pr_test_job = f"dlc-pr-{test_type}-test"
            images_str = " ".join(images)
            # Maintaining separate codebuild project for graviton sanity test
            if "graviton" in images_str and test_type == "sanity":
                pr_test_job += "-graviton"
            if is_test_job_enabled(test_type) and is_test_job_implemented_for_framework(
                images_str, test_type
            ):
                run_test_job(commit, pr_test_job, images_str)

            # Trigger sagemaker local test jobs when there are changes in sagemaker_tests
            # sagemaker local test is not supported in benchmark dev mode
            if (
                test_type == "sagemaker"
                and not config.is_benchmark_mode_enabled()
                and config.is_sm_local_test_enabled()
            ):
                test_job = f"dlc-pr-{test_type}-local-test"
                run_test_job(commit, test_job, images_str)


if __name__ == "__main__":
    main()
