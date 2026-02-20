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
from __future__ import absolute_import

import logging
import os
import sys

import boto3
import pytest
from sagemaker import LocalSession, Session

from . import get_ecr_registry, get_efa_test_instance_type

logger = logging.getLogger(__name__)


dir_path = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption("--build-image", "-D", action="store_true")
    parser.addoption("--build-base-image", "-B", action="store_true")
    parser.addoption("--aws-id")
    parser.addoption("--instance-type")
    parser.addoption("--docker-base-name", default="pytorch")
    parser.addoption("--region", default="us-west-2")
    parser.addoption("--framework-version", default="")
    parser.addoption(
        "--py-version",
        choices=["312"],
        default=str(sys.version_info.major),
    )
    parser.addoption("--processor", choices=["gpu"], default="gpu")
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption("--tag", default=None)
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


def pytest_configure(config):
    config.addinivalue_line("markers", "efa(): explicitly mark to run efa tests")


def pytest_runtest_setup(item):
    efa_tests = [mark for mark in item.iter_markers(name="efa")]
    if item.config.getoption("--efa") and not efa_tests:
        pytest.skip("Skipping non-efa tests due to --efa flag")
    elif not item.config.getoption("--efa") and efa_tests:
        pytest.skip("Skipping efa tests because --efa flag is missing")


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        print(f"item {item}")
        for marker in item.iter_markers(name="team"):
            print(f"item {marker}")
            team_name = marker.args[0]
            item.user_properties.append(("team_marker", team_name))
            print(f"item.user_properties {item.user_properties}")

    if config.getoption("--generate-coverage-doc"):
        from test.test_utils.test_reporting import TestReportGenerator

        report_generator = TestReportGenerator(items, is_sagemaker=True)
        report_generator.generate_coverage_doc(framework="pytorch", job_type="training")


@pytest.fixture(scope="session", name="docker_base_name")
def fixture_docker_base_name(request):
    return request.config.getoption("--docker-base-name")


@pytest.fixture(scope="session", name="region")
def fixture_region(request):
    return request.config.getoption("--region")


@pytest.fixture(scope="session", name="framework_version")
def fixture_framework_version(request):
    return request.config.getoption("--framework-version")


@pytest.fixture(scope="session", name="py_version")
def fixture_py_version(request):
    return "py{}".format(int(request.config.getoption("--py-version")))


@pytest.fixture(scope="session", name="processor")
def fixture_processor(request):
    return request.config.getoption("--processor")


@pytest.fixture(scope="session", name="sagemaker_regions")
def fixture_sagemaker_regions(request):
    sagemaker_regions = request.config.getoption("--sagemaker-regions")
    return sagemaker_regions.split(",")


@pytest.fixture(scope="session", name="tag")
def fixture_tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption("--tag")
    default_tag = "{}-{}-{}".format(framework_version, processor, py_version)
    return provided_tag if provided_tag else default_tag


@pytest.fixture(scope="session", name="docker_image")
def fixture_docker_image(docker_base_name, tag):
    return "{}:{}".format(docker_base_name, tag)


@pytest.fixture(scope="session", name="sagemaker_session")
def fixture_sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(name="efa_instance_type")
def fixture_efa_instance_type(request):
    try:
        return request.param
    except AttributeError:
        return get_efa_test_instance_type(default=["ml.p4d.24xlarge"])[0]


@pytest.fixture(scope="session", name="sagemaker_local_session")
def fixture_sagemaker_local_session(region):
    return LocalSession(boto_session=boto3.Session(region_name=region))


@pytest.fixture(name="aws_id", scope="session")
def fixture_aws_id(request):
    return request.config.getoption("--aws-id")


@pytest.fixture(name="instance_type", scope="session")
def fixture_instance_type(request, processor):
    provided_instance_type = request.config.getoption("--instance-type")
    default_instance_type = "local" if processor == "cpu" else "local_gpu"
    return provided_instance_type or default_instance_type


@pytest.fixture(name="docker_registry", scope="session")
def fixture_docker_registry(aws_id, region):
    return get_ecr_registry(aws_id, region)


@pytest.fixture(name="ecr_image", scope="session")
def fixture_ecr_image(docker_registry, docker_base_name, tag):
    return "{}/{}:{}".format(docker_registry, docker_base_name, tag)
