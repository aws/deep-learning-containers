# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import logging
import os

import boto3
import pytest
from sagemaker import LocalSession, Session
from sagemaker.mxnet import MXNet

from .integration import NO_P2_REGIONS

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption('--docker-base-name', default='preprod-mxnet')
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--instance-count', default='1,2', choices=['1', '2', '1,2'])
    parser.addoption('--framework-version', default=MXNet.LATEST_VERSION)
    parser.addoption('--py-version', default='3', choices=['2', '3', '2,3'])
    parser.addoption('--processor', default='cpu', choices=['gpu', 'cpu', 'cpu,gpu'])
    parser.addoption('--aws-id', default=None)
    parser.addoption('--instance-type', default=None)
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption('--tag', default=None)


def pytest_generate_tests(metafunc):
    if 'instance_count' in metafunc.fixturenames:
        ic_params = [int(x) for x in metafunc.config.getoption('--instance-count').split(',')]
        metafunc.parametrize('instance_count', ic_params, scope='session')

    if 'py_version' in metafunc.fixturenames:
        py_version_params = ['py' + v for v in metafunc.config.getoption('--py-version').split(',')]
        metafunc.parametrize('py_version', py_version_params, scope='session')

    if 'processor' in metafunc.fixturenames:
        processor_params = metafunc.config.getoption('--processor').split(',')
        metafunc.parametrize('processor', processor_params, scope='session')


@pytest.fixture(scope='session')
def docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session')
def aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(scope='session')
def tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-{}'.format(framework_version, processor, py_version)
    return provided_tag if provided_tag is not None else default_tag


@pytest.fixture(scope='session')
def instance_type(request, processor):
    provided_instance_type = request.config.getoption('--instance-type')
    default_instance_type = 'ml.c4.xlarge' if processor == 'cpu' else 'ml.p2.xlarge'
    return provided_instance_type if provided_instance_type is not None else default_instance_type


@pytest.fixture(scope='session')
def docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture(scope='session')
def ecr_image(aws_id, docker_base_name, tag, region):
    return '{}.dkr.ecr.{}.amazonaws.com/{}:{}'.format(aws_id, region, docker_base_name, tag)


@pytest.fixture(scope='session')
def sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def sagemaker_local_session(region):
    return LocalSession(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session')
def local_instance_type(processor):
    return 'local' if processor == 'cpu' else 'local_gpu'


@pytest.fixture(autouse=True)
def skip_test_in_region(request, region):
    if request.node.get_closest_marker('skip_test_in_region'):
        if region == 'me-south-1':
            pytest.skip('Skipping SageMaker test in region {}'.format(region))


@pytest.fixture(autouse=True)
def skip_gpu_instance_restricted_regions(region, instance_type):
    if region in NO_P2_REGIONS and instance_type.startswith('ml.p2'):
        pytest.skip('Skipping GPU test in region {} to avoid insufficient capacity'.format(region))


@pytest.fixture(autouse=True)
def skip_py2_containers(request, tag):
    if request.node.get_closest_marker('skip_py2_containers'):
        if 'py2' in tag:
            pytest.skip('Skipping python2 container with tag {}'.format(tag))
