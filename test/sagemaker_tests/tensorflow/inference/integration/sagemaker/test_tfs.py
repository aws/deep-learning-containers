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

import pytest

import util

NON_P3_REGIONS = ['ap-southeast-1', 'ap-southeast-2', 'ap-south-1',
                  'ca-central-1', 'eu-central-1', 'eu-west-2', 'us-west-1']


@pytest.fixture(params=os.environ['TEST_VERSIONS'].split(','))
def version(request):
    return request.param


@pytest.fixture(scope='session')
def repo(request):
    return request.config.getoption('--repo') or 'sagemaker-tensorflow-serving'


@pytest.fixture
def tag(request, version, instance_type):
    if request.config.getoption('--tag'):
        return request.config.getoption('--tag')

    arch = 'gpu' if instance_type.startswith('ml.p') else 'cpu'
    return f'{version}-{arch}'


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ['TEST_INSTANCE_TYPES'].split(','))
def instance_type(request, region):
    return request.param


@pytest.fixture(scope='module')
def accelerator_type():
    return None


@pytest.fixture(scope='session')
def tfs_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       'test/data/tfs-model.tar.gz')


@pytest.fixture(scope='session')
def python_model_with_requirements(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       'test/data/python-with-requirements.tar.gz')


@pytest.fixture(scope='session')
def python_model_with_lib(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       'test/data/python-with-lib.tar.gz')


def test_tfs_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {'instances': [1.0, 2.0, 5.0]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_model,
                                    image_uri, instance_type, accelerator_type, input_data)


def test_batch_transform(region, boto_session, sagemaker_client,
                         model_name, tfs_model, image_uri,
                         instance_type):
    results = util.run_batch_transform_job(region=region,
                                           boto_session=boto_session,
                                           model_data=tfs_model,
                                           image_uri=image_uri,
                                           model_name=model_name,
                                           sagemaker_client=sagemaker_client,
                                           instance_type=instance_type)
    assert len(results) == 10
    for r in results:
        assert r == [3.5, 4.0, 5.5]


def test_python_model_with_requirements(boto_session, sagemaker_client,
                                        sagemaker_runtime_client, model_name,
                                        python_model_with_requirements, image_uri, instance_type,
                                        accelerator_type):

    if 'p3' in instance_type:
        pytest.skip('skip for p3 instance')

    # the python service needs to transform this to get a valid prediction
    input_data = {'x': [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name,
                                                  python_model_with_requirements, image_uri,
                                                  instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data['python'] is True
    assert output_data['pillow'] == '6.0.0'


def test_python_model_with_lib(boto_session, sagemaker_client,
                               sagemaker_runtime_client, model_name, python_model_with_lib,
                               image_uri, instance_type, accelerator_type):

    if 'p3' in instance_type:
        pytest.skip('skip for p3 instance')

    # the python service needs to transform this to get a valid prediction
    input_data = {'x': [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name, python_model_with_lib,
                                                  image_uri, instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data['python'] is True
    assert output_data['dummy_module'] == '0.1'
