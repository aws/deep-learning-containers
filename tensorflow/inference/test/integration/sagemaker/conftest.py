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
import os
import random
import time

import boto3
import pytest

# these regions have some p2 and p3 instances, but not enough for automated testing
NO_P2_REGIONS = [
    'ca-central-1',
    'eu-central-1',
    'eu-west-2',
    'us-west-1',
    'eu-west-3',
    'eu-north-1',
    'sa-east-1',
    'ap-east-1',
    'me-south-1'
]
NO_P3_REGIONS = [
    'ap-southeast-1',
    'ap-southeast-2',
    'ap-south-1',
    'ca-central-1',
    'eu-central-1',
    'eu-west-2',
    'us-west-1',
    'eu-west-3',
    'eu-north-1',
    'sa-east-1',
    'ap-east-1',
    'me-south-1'
]


def pytest_addoption(parser):
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--registry')
    parser.addoption('--repo')
    parser.addoption('--versions')
    parser.addoption('--instance-types')
    parser.addoption('--accelerator-type')
    parser.addoption('--tag')


def pytest_configure(config):
    os.environ['TEST_REGION'] = config.getoption('--region')
    os.environ['TEST_VERSIONS'] = config.getoption('--versions') or '1.11.1,1.12.0,1.13.0'
    os.environ['TEST_INSTANCE_TYPES'] = (config.getoption('--instance-types') or
                                         'ml.m5.xlarge,ml.p2.xlarge')

    os.environ['TEST_EI_VERSIONS'] = config.getoption('--versions') or '1.11,1.12'
    os.environ['TEST_EI_INSTANCE_TYPES'] = (config.getoption('--instance-types') or
                                            'ml.m5.xlarge')

    if config.getoption('--tag'):
        os.environ['TEST_VERSIONS'] = config.getoption('--tag')
        os.environ['TEST_EI_VERSIONS'] = config.getoption('--tag')


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def registry(request, region):
    if request.config.getoption('--registry'):
        return request.config.getoption('--registry')

    sts = boto3.client(
        'sts',
        region_name=region,
        endpoint_url='https://sts.{}.amazonaws.com'.format(region)
    )
    return sts.get_caller_identity()['Account']


@pytest.fixture(scope='session')
def boto_session(region):
    return boto3.Session(region_name=region)


@pytest.fixture(scope='session')
def sagemaker_client(boto_session):
    return boto_session.client('sagemaker')


@pytest.fixture(scope='session')
def sagemaker_runtime_client(boto_session):
    return boto_session.client('runtime.sagemaker')


def unique_name_from_base(base, max_length=63):
    unique = '%04x' % random.randrange(16 ** 4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return '{}-{}-{}'.format(trimmed, ts, unique)


@pytest.fixture
def model_name():
    return unique_name_from_base('test-tfs')


@pytest.fixture(autouse=True)
def skip_gpu_instance_restricted_regions(region, instance_type):
    if (region in NO_P2_REGIONS and instance_type.startswith('ml.p2')) or \
            (region in NO_P3_REGIONS and instance_type.startswith('ml.p3')):
        pytest.skip('Skipping GPU test in region {}'.format(region))
