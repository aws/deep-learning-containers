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

    pr_num = os.getenv("CODEBUILD_SOURCE_VERSION")
    env_overrides.extend(
        [
            {"name": "DLC_IMAGES", "value": images_str, "type": "PLAINTEXT"},
            {"name": "PR_NUMBER", "value": pr_num, "type": "PLAINTEXT"},
            # USE_SCHEDULER is passed as an env variable here because it is more convenient to set this in
            # dlc_developer_config, compared to having another config file under dlc/tests/.
            {
                "name": "USE_SCHEDULER",
                "value": str(config.is_scheduler_enabled()),
                "type": "PLAINTEXT",
            },
            {
                "name": "DISABLE_EFA_TESTS",
                "value": str(not config.are_efa_tests_enabled()),
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
            if "graviton" in images_str:
                pr_test_job += "graviton"
            if is_test_job_enabled(test_type):
                if "huggingface" in images_str and test_type in [
                    constants.EC2_TESTS,
                    constants.ECS_TESTS,
                    constants.EKS_TESTS,
                ]:
                    LOGGER.debug(f"Skipping huggingface {test_type} test")
                    continue
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
