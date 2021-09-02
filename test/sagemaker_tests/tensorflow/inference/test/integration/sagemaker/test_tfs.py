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

from ..sagemaker import util

NON_P3_REGIONS = ["ap-southeast-1", "ap-southeast-2", "ap-south-1",
                  "ca-central-1", "eu-central-1", "eu-west-2", "us-west-1"]

@pytest.fixture(name='use_gpu')
def fixture_use_gpu(processor):
    return processor == 'gpu'


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

@pytest.fixture(scope="session")
def docker_base_name(request):
    return request.config.getoption("--docker-base-name") or "sagemaker-tensorflow-serving"


@pytest.fixture(scope="session")
def tfs_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/tfs-model.tar.gz")


@pytest.fixture(scope="session")
def python_model_with_requirements(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/python-with-requirements.tar.gz")


@pytest.fixture(scope="session")
def python_model_with_lib(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/python-with-lib.tar.gz")


@pytest.mark.model("unknown_model")
@pytest.mark.cpu_test
@pytest.mark.release_test
def test_tfs_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_model,
                                    image_uri, instance_type, accelerator_type, input_data)


@pytest.mark.integration("batch_transform")
@pytest.mark.model("unknown_model")
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


@pytest.mark.model("unknown_model")
def test_python_model_with_requirements(boto_session, sagemaker_client,
                                        sagemaker_runtime_client, model_name,
                                        python_model_with_requirements, image_uri, instance_type,
                                        accelerator_type):

    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    # the python service needs to transform this to get a valid prediction
    input_data = {"x": [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name,
                                                  python_model_with_requirements, image_uri,
                                                  instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data["python"] is True
    assert output_data["pillow"] == "6.0.0"


@pytest.mark.model("unknown_model")
def test_python_model_with_lib(boto_session, sagemaker_client,
                               sagemaker_runtime_client, model_name, python_model_with_lib,
                               image_uri, instance_type, accelerator_type):

    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    # the python service needs to transform this to get a valid prediction
    input_data = {"x": [1.0, 2.0, 5.0]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name, python_model_with_lib,
                                                  image_uri, instance_type, accelerator_type, input_data)

    # python service adds this to tfs response
    assert output_data["python"] is True
    assert output_data["dummy_module"] == "0.1"
