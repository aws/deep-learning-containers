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

    env_overrides.append(
        {"name": "DLC_IMAGES", "value": images_str, "type": "PLAINTEXT"}
    )
    LOGGER.debug(f"env_overrides dict: {env_overrides}")

    client = boto3.client("codebuild")
    return client.start_build(
        projectName=codebuild_project,
        environmentVariablesOverride=env_overrides,
        sourceVersion=commit,
    )


def main():
    build_context = os.getenv("BUILD_CONTEXT")
    if build_context != "PR":
        LOGGER.info(
            f"Not triggering test jobs from boto3, as BUILD_CONTEXT is {build_context}"
        )
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
            pr_test_job = f"dlc-{test_type}-test"
            images_str = " ".join(images)
            run_test_job(commit, pr_test_job, images_str)


if __name__ == "__main__":
    main()
