# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import json
import os
import random
import time

import boto3
import pytest

from botocore.config import Config
from botocore.exceptions import ClientError


# these regions have some p2 and p3 instances, but not enough for automated testing
NO_P2_REGIONS = [
    "ap-northeast-3",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]
NO_P3_REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]
NO_P4_REGIONS = [
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]


def pytest_addoption(parser):
    parser.addoption("--region", default="us-west-2")
    parser.addoption("--registry")
    parser.addoption("--repo")
    parser.addoption("--versions")
    parser.addoption("--instance-types")
    parser.addoption("--processor")
    parser.addoption("--accelerator-type")
    parser.addoption("--tag")
    parser.addoption(
        "--generate-coverage-doc",
        default=False,
        action="store_true",
        help="use this option to generate test coverage doc",
    )
    parser.addoption(
        "--efa",
        action="store_true",
        default=False,
        help="Run only efa tests",
    )
    parser.addoption("--sagemaker-regions", default="us-west-2")


def pytest_runtest_setup(item):
    if item.config.getoption("--efa"):
        efa_tests = [mark for mark in item.iter_markers(name="efa")]
        if not efa_tests:
            pytest.skip("Skipping non-efa tests")


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--generate-coverage-doc"):
        from test.test_utils.test_reporting import TestReportGenerator

        report_generator = TestReportGenerator(items, is_sagemaker=True)
        report_generator.generate_coverage_doc(framework="tensorflow", job_type="inference")


def pytest_configure(config):
    os.environ["TEST_REGION"] = config.getoption("--region")
    os.environ["TEST_VERSIONS"] = config.getoption("--versions") or "1.11.1,1.12.0,1.13.0"
    os.environ["TEST_INSTANCE_TYPES"] = (
        config.getoption("--instance-types") or "ml.m5.xlarge,ml.p2.xlarge"
    )

    os.environ["TEST_EI_VERSIONS"] = config.getoption("--versions") or "1.11,1.12"
    os.environ["TEST_EI_INSTANCE_TYPES"] = config.getoption("--instance-types") or "ml.m5.xlarge"

    if config.getoption("--tag"):
        os.environ["TEST_VERSIONS"] = config.getoption("--tag")
        os.environ["TEST_EI_VERSIONS"] = config.getoption("--tag")
    config.addinivalue_line("markers", "efa(): explicitly mark to run efa tests")


# Nightly fixtures
@pytest.fixture(scope="session")
def feature_aws_framework_present():
    pass


@pytest.fixture(scope="session")
def region(request):
    return request.config.getoption("--region")


@pytest.fixture(scope="session", name="sagemaker_regions")
def sagemaker_regions(request):
    sagemaker_regions = request.config.getoption("--sagemaker-regions")
    return sagemaker_regions.split(",")


@pytest.fixture(scope="session")
def registry(request, region):
    if request.config.getoption("--registry"):
        return request.config.getoption("--registry")

    domain_suffix = ".cn" if region in ("cn-north-1", "cn-northwest-1") else ""
    sts_regional_endpoint = "https://sts.{}.amazonaws.com{}".format(region, domain_suffix)

    sts = boto3.client("sts", region_name=region, endpoint_url=sts_regional_endpoint)
    return sts.get_caller_identity()["Account"]


@pytest.fixture(scope="session")
def boto_session(region):
    return boto3.Session(region_name=region)


@pytest.fixture(scope="session")
def sagemaker_client(boto_session):
    return boto_session.client("sagemaker", config=Config(retries={"max_attempts": 10}))


@pytest.fixture(scope="session")
def sagemaker_runtime_client(boto_session):
    return boto_session.client("runtime.sagemaker", config=Config(retries={"max_attempts": 10}))


def unique_name_from_base(base, max_length=63):
    unique = "%04x" % random.randrange(16**4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return "{}-{}-{}".format(trimmed, ts, unique)


@pytest.fixture
def model_name():
    return unique_name_from_base("test-tfs")


@pytest.fixture(autouse=True)
def skip_gpu_instance_restricted_regions(region, instance_type):
    if (
        (region in NO_P2_REGIONS and instance_type.startswith("ml.p2"))
        or (region in NO_P3_REGIONS and instance_type.startswith("ml.p3"))
        or (region in NO_P4_REGIONS and instance_type.startswith("ml.p4"))
    ):
        pytest.skip("Skipping GPU test in region {}".format(region))


@pytest.fixture(autouse=True)
def skip_by_device_type(request, instance_type):
    is_gpu = instance_type.lstrip("ml.")[0] in ["g", "p"]

    # Skip a neuron(x) test that's not on an neuron instance or a test which
    # uses a neuron instance and is not a neuron(x) test
    is_neuron_test = request.node.get_closest_marker("neuron_test") is not None
    is_neuron_instance = instance_type.startswith("ml.inf1")
    if is_neuron_test != is_neuron_instance:
        pytest.skip("Skipping because running on '{}' instance".format(instance_type))

    is_neuronx_test = request.node.get_closest_marker("neuronx_test") is not None
    is_neuronx_instance = instance_type.startswith("ml.trn1")
    if is_neuronx_test != is_neuronx_instance:
        pytest.skip("Skipping because running on '{}' instance".format(instance_type))

    if (request.node.get_closest_marker("skip_gpu") and is_gpu) or (
        request.node.get_closest_marker("skip_cpu") and not is_gpu
    ):
        pytest.skip('Skipping because running on "{}" instance'.format(instance_type))


def _get_remote_override_flags():
    try:
        s3_client = boto3.client("s3")
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity().get("Account")
        result = s3_client.get_object(
            Bucket=f"dlc-cicd-helper-{account_id}", Key="override_tests_flags.json"
        )
        json_content = json.loads(result["Body"].read().decode("utf-8"))
    except ClientError as e:
        print("ClientError when performing S3/STS operation: {}".format(e))
        json_content = {}
    return json_content


def _is_test_disabled(test_name, build_name, version):
    """
    Expected format of remote_override_flags:
    {
        "CB Project Name for Test Type A": {
            "CodeBuild Resolved Source Version": ["test_type_A_test_function_1", "test_type_A_test_function_2"]
        },
        "CB Project Name for Test Type B": {
            "CodeBuild Resolved Source Version": ["test_type_B_test_function_1", "test_type_B_test_function_2"]
        }
    }

    :param test_name: str Test Function node name (includes parametrized values in string)
    :param build_name: str Build Project name of current execution
    :param version: str Source Version of current execution
    :return: bool True if test is disabled as per remote override, False otherwise
    """
    remote_override_flags = _get_remote_override_flags()
    remote_override_build = remote_override_flags.get(build_name, {})
    if version in remote_override_build:
        return not remote_override_build[version] or any(
            [test_keyword in test_name for test_keyword in remote_override_build[version]]
        )
    return False


@pytest.fixture(autouse=True)
def disable_test(request):
    test_name = request.node.name
    # We do not have a regex pattern to find CB name, which means we must resort to string splitting
    build_arn = os.getenv("CODEBUILD_BUILD_ARN")
    build_name = build_arn.split("/")[-1].split(":")[0] if build_arn else None
    version = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")

    if build_name and version and _is_test_disabled(test_name, build_name, version):
        pytest.skip(f"Skipping {test_name} test because it has been disabled.")


@pytest.fixture(autouse=True)
def skip_test_successfully_executed_before(request):
    """
    "cache/lastfailed" contains information about failed tests only. We're running SM tests in separate threads for each image.
    So when we retry SM tests, successfully executed tests executed again because pytest doesn't have that info in /.cache.
    But the flag "--last-failed-no-failures all" requires pytest to execute all the available tests.
    The only sign that a test passed last time - lastfailed file exists and the test name isn't in that file.
    The method checks whether lastfailed file exists and the test name is not in it.
    """
    test_name = request.node.name
    lastfailed = request.config.cache.get("cache/lastfailed", None)

    if lastfailed is not None and not any(
        test_name in failed_test_name for failed_test_name in lastfailed.keys()
    ):
        pytest.skip(f"Skipping {test_name} because it was successfully executed for this commit")
