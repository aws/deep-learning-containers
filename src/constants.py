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

# Environment settings
FRAMEWORKS = {"mxnet", "tensorflow", "pytorch"}
DEVICE_TYPES = {"cpu", "gpu"}
IMAGE_TYPES = {"training", "inference"}
PYTHON_VERSIONS = {"py2", "py3"}
ALL = "all"

# Function Status Codes
SUCCESS = 0
FAIL = 1
NOT_BUILT = -1

# Left and right padding between text and margins in output
PADDING = 1

# Docker connections
DOCKER_URL = "unix://var/run/docker.sock"

STATUS_MESSAGE = {SUCCESS: "Success", FAIL: "Failed", NOT_BUILT: "Not Built"}

BUILD_CONTEXT = os.environ.get("BUILD_CONTEXT", "DEV")

METRICS_NAMESPACE = "dlc-metrics-to-be-deleted"

# Logging level
INFO = 1
ERROR = 2
DEBUG = 3

# Repository prefix
MAINLINE_REPO_PREFIX = "beta-"
NIGHTLY_REPO_PREFIX = "nightly-"
PR_REPO_PREFIX = "pr-"

# Env variables for the code build PR jobs
JOB_FRAMEWORK = os.environ.get("FRAMEWORK")
JOB_FRAMEWORK_VERSION = os.environ.get("VERSION")

# Test environment file
TEST_ENV_PATH = os.path.join(os.path.expanduser("~"), "testenv.json")
# Test images for all test types (sm,ecs,eks,ec2) to pass on to code build jobs
TEST_TYPE_IMAGES_PATH = os.path.join(os.path.expanduser("~"), "test_type_images.json")

# Test types for running code build test jobs
SAGEMAKER_TESTS = "sagemaker"
SANITY_TESTS = "sanity"
EC2_TESTS = "ec2"
ECS_TESTS = "ecs"
EKS_TESTS = "eks"
ALL_TESTS = ["sagemaker", "ec2", "eks", "ecs"]
