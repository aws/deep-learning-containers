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
from __future__ import absolute_import

import boto3
import os
import logging
import platform
import pytest
import shutil
import sys
import tempfile

from sagemaker import LocalSession, Session
from sagemaker.pytorch import PyTorch

from .utils import get_ecr_registry
from .utils import image_utils

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)


dir_path = os.path.dirname(os.path.realpath(__file__))

NO_P2_REGIONS = ['ap-east-1', 'ap-northeast-3', 'ap-southeast-2', 'ca-central-1', 'eu-central-1', 'eu-north-1',
                 'eu-west-2', 'eu-west-3', 'us-west-1', 'sa-east-1', 'me-south-1', 'cn-northwest-1']
NO_P3_REGIONS = ['ap-east-1', 'ap-northeast-3', 'ap-southeast-1', 'ap-southeast-2', 'ap-south-1', 'ca-central-1',
                 'eu-central-1', 'eu-north-1', 'eu-west-2', 'eu-west-3', 'sa-east-1', 'us-west-1', 'me-south-1',
                 'cn-northwest-1']


def pytest_addoption(parser):
    parser.addoption('--build-image', '-D', action='store_true')
    parser.addoption('--build-base-image', '-B', action='store_true')
    parser.addoption('--aws-id')
    parser.addoption('--instance-type')
    parser.addoption('--accelerator-type', default=None)
    parser.addoption('--docker-base-name', default='pytorch')
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default=PyTorch.LATEST_VERSION)
    parser.addoption('--py-version', choices=['2', '3'], default=str(sys.version_info.major))
    # Processor is still "cpu" for EIA tests
    parser.addoption('--processor', choices=['gpu', 'cpu', 'eia'], default='cpu')
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption('--tag', default=None)
    parser.addoption('--generate-coverage-doc', default=False, action='store_true',
                     help='use this option to generate test coverage doc')


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--generate-coverage-doc"):
        from test.test_utils.test_reporting import TestReportGenerator
        report_generator = TestReportGenerator(items, is_sagemaker=True)
        report_generator.generate_coverage_doc(framework="pytorch", job_type="inference")


@pytest.fixture(scope='session', name='docker_base_name')
def fixture_docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session', name='region')
def fixture_region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session', name='framework_version')
def fixture_framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session', name='py_version')
def fixture_py_version(request):
    return 'py{}'.format(int(request.config.getoption('--py-version')))


@pytest.fixture(scope='session', name='processor')
def fixture_processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session', name='tag')
def fixture_tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-{}'.format(framework_version, processor, py_version)
    return provided_tag if provided_tag else default_tag


@pytest.fixture(scope='session', name='docker_image')
def fixture_docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session', name='use_gpu')
def fixture_use_gpu(processor):
    return processor == 'gpu'


@pytest.fixture(scope='session', name='build_base_image', autouse=True)
def fixture_build_base_image(request, framework_version, py_version, processor, tag, docker_base_name):
    build_base_image = request.config.getoption('--build-base-image')
    if build_base_image:
        return image_utils.build_base_image(framework_name=docker_base_name,
                                            framework_version=framework_version,
                                            py_version=py_version,
                                            base_image_tag=tag,
                                            processor=processor,
                                            cwd=os.path.join(dir_path, '..'))

    return tag


@pytest.fixture(scope='session', name='sagemaker_session')
def fixture_sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session', name='sagemaker_local_session')
def fixture_sagemaker_local_session(region):
    return LocalSession(boto_session=boto3.Session(region_name=region))


@pytest.fixture(name='aws_id', scope='session')
def fixture_aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(name='instance_type', scope='session')
def fixture_instance_type(request, processor):
    provided_instance_type = request.config.getoption('--instance-type')
    default_instance_type = 'local' if processor == 'cpu' else 'local_gpu'
    return provided_instance_type or default_instance_type


@pytest.fixture(name='accelerator_type', scope='session')
def fixture_accelerator_type(request):
    return request.config.getoption('--accelerator-type')


@pytest.fixture(name='docker_registry', scope='session')
def fixture_docker_registry(aws_id, region):
    return get_ecr_registry(aws_id, region)


@pytest.fixture(name='ecr_image', scope='session')
def fixture_ecr_image(docker_registry, docker_base_name, tag):
    return '{}/{}:{}'.format(docker_registry, docker_base_name, tag)


@pytest.fixture(autouse=True)
def skip_by_device_type(request, use_gpu, instance_type, accelerator_type):
    is_gpu = use_gpu or instance_type[3] in ['g', 'p']
    is_eia = accelerator_type is not None

    # Separate out cases for clearer logic.
    # When running GPU test, skip CPU test. When running CPU test, skip GPU test.
    if (request.node.get_closest_marker('gpu_test') and not is_gpu) or \
            (request.node.get_closest_marker('cpu_test') and is_gpu):
        pytest.skip('Skipping because running on \'{}\' instance'.format(instance_type))

    # When running EIA test, skip the CPU and GPU functions
    elif (request.node.get_closest_marker('gpu_test') or request.node.get_closest_marker('cpu_test')) and is_eia:
        pytest.skip('Skipping because running on \'{}\' instance'.format(instance_type))

    # When running CPU or GPU test, skip EIA test.
    elif request.node.get_closest_marker('eia_test') and not is_eia:
        pytest.skip('Skipping because running on \'{}\' instance'.format(instance_type))


@pytest.fixture(autouse=True)
def skip_by_py_version(request, py_version):
    if request.node.get_closest_marker('skip_py2') and py_version != 'py3':
        pytest.skip('Skipping the test because Python 2 is not supported.')


@pytest.fixture(autouse=True)
def skip_gpu_instance_restricted_regions(region, instance_type):
    if (region in NO_P2_REGIONS and instance_type.startswith('ml.p2')) \
       or (region in NO_P3_REGIONS and instance_type.startswith('ml.p3')):
        pytest.skip('Skipping GPU test in region {}'.format(region))


@pytest.fixture(autouse=True)
def skip_gpu_py2(request, use_gpu, instance_type, py_version, framework_version):
    is_gpu = use_gpu or instance_type[3] in ['g', 'p']
    if request.node.get_closest_marker('skip_gpu_py2') and is_gpu and py_version != 'py3' \
            and framework_version == '1.4.0':
        pytest.skip('Skipping the test until mms issue resolved.')
