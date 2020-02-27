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

import pytest

import util

EI_SUPPORTED_REGIONS = ['us-east-1', 'us-east-2', 'us-west-2',
                        'eu-west-1', 'ap-northeast-1', 'ap-northeast-2']


@pytest.fixture(params=os.environ['TEST_EI_VERSIONS'].split(','))
def version(request):
    return request.param


@pytest.fixture
def repo(request):
    return request.config.getoption('--repo') or 'sagemaker-tensorflow-serving-eia'


@pytest.fixture
def tag(request, version):
    return request.config.getoption('--tag') or f'{version}-cpu'


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ['TEST_EI_INSTANCE_TYPES'].split(','))
def instance_type(request, region):
    return request.param


@pytest.fixture(scope='module')
def accelerator_type(request):
    return request.config.getoption('--accelerator-type') or 'ml.eia1.medium'


@pytest.fixture(scope='session')
def model_data(region):
    return ('s3://sagemaker-sample-data-{}/tensorflow/model'
            '/resnet/resnet_50_v2_fp32_NCHW.tar.gz').format(region)


@pytest.fixture
def input_data():
    return {'instances': [[[[random.random() for _ in range(3)] for _ in range(3)]]]}


@pytest.fixture
def skip_if_no_accelerator(accelerator_type):
    if accelerator_type is None:
        pytest.skip('Skipping because accelerator type was not provided')


@pytest.fixture
def skip_if_non_supported_ei_region(region):
    if region not in EI_SUPPORTED_REGIONS:
        pytest.skip('EI is not supported in {}'.format(region))


@pytest.mark.skip_if_non_supported_ei_region()
@pytest.mark.skip_if_no_accelerator()
def test_invoke_endpoint(boto_session, sagemaker_client, sagemaker_runtime_client,
                         model_name, model_data, image_uri, instance_type, accelerator_type,
                         input_data):
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, model_data, image_uri,
                                    instance_type, accelerator_type, input_data)
