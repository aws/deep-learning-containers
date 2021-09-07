# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import boto3
import pytest

from botocore.exceptions import ClientError


TFS_DOCKER_BASE_NAME = 'sagemaker-tensorflow-serving'


def pytest_addoption(parser):
    parser.addoption('--docker-base-name', default=TFS_DOCKER_BASE_NAME)
    parser.addoption('--framework-version', required=True)
    parser.addoption('--processor', default='cpu', choices=['cpu', 'gpu'])
    parser.addoption('--aws-id', default=None)
    parser.addoption('--tag')
    parser.addoption('--generate-coverage-doc', default=False, action='store_true',
                     help='use this option to generate test coverage doc')
    parser.addoption('--sagemaker-regions')

def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--generate-coverage-doc"):
        from test.test_utils.test_reporting import TestReportGenerator
        report_generator = TestReportGenerator(items, is_sagemaker=True)
        report_generator.generate_coverage_doc(framework="tensorflow", job_type="inference")


@pytest.fixture(scope='module')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='module')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='module')
def processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='module')
def runtime_config(request, processor):
    if processor == 'gpu':
        return '--runtime=nvidia '
    else:
        return ''


@pytest.fixture(scope='module')
def tag(request, framework_version, processor):
    image_tag = request.config.getoption('--tag')
    if not image_tag:
        image_tag = '{}-{}'.format(framework_version, processor)
    return image_tag


@pytest.fixture(autouse=True)
def skip_by_device_type(request, processor):
    is_gpu = processor == 'gpu'
    if (request.node.get_closest_marker('skip_gpu') and is_gpu) or \
            (request.node.get_closest_marker('skip_cpu') and not is_gpu):
        pytest.skip('Skipping because running on \'{}\' instance'.format(processor))


def _get_remote_override_flags():
    try:
        s3_client = boto3.client('s3')
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity().get('Account')
        result = s3_client.get_object(Bucket=f"dlc-cicd-helper-{account_id}", Key="override_tests_flags.json")
        json_content = json.loads(result["Body"].read().decode('utf-8'))
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
        return (
            not remote_override_build[version]
            or any([test_keyword in test_name for test_keyword in remote_override_build[version]])
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
