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
import json

import boto3

import constants


def run_test_job(commit, codebuild_project):
    test_env_file = constants.TEST_ENV
    if not os.path.exists(test_env_file):
        raise FileNotFoundError(f"{test_env_file} not found. This is required to set test environment variables"
                                f" for test jobs. Failing the build.")

    with open(test_env_file) as test_env_file:
        env_overrides = json.load(test_env_file)

    # Make sure DLC_IMAGES exists. If not, don't execute job.
    images_present = False
    for override in env_overrides:
        if override.get('name') == "DLC_IMAGES" and override.get('value', '').strip():
            images_present = True
            break

    if not images_present:
        print(f"Skipping test {codebuild_project} as no images were built.")
        return

    client = boto3.client("codebuild")
    return client.start_build(
        projectName=codebuild_project,
        environmentVariablesOverride=env_overrides,
        sourceVersion=commit,
    )


def main():
    build_context = os.getenv("BUILD_CONTEXT")
    if build_context != "PR":
        print(f"Not triggering test jobs from boto3, as BUILD_CONTEXT is {build_context}")
        return

    # Start sanity test job
    commit = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    run_test_job(commit, "dlc-sanity-test")
    run_test_job(commit, "dlc-sagemaker-test")


if __name__ == "__main__":
    main()
