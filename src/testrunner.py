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
    with open(constants.TEST_ENV) as test_env_file:
        env_overrides = json.load(test_env_file)

    client = boto3.client("codebuild")
    return client.start_build(
        projectName=codebuild_project,
        environmentVariablesOverride=env_overrides,
        sourceVersion=commit
    )


def main():
    # Start sanity test job
    commit = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    run_test_job(commit, "dlc-sanity-test")


if __name__ == "__main__":
    main()
