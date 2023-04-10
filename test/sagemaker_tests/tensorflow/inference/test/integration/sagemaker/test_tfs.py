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


@pytest.fixture(params=os.environ["TEST_VERSIONS"].split(","))
def version(request):
    return request.param


@pytest.fixture(scope="session")
def repo(request):
    return request.config.getoption("--repo") or "sagemaker-tensorflow-serving"


@pytest.fixture
def processor(request, instance_type):
    return request.config.getoption("--processor") or (
        "gpu"
        if instance_type.startswith("ml.p") or instance_type.startswith("ml.g")
        else "cpu"
    )


@pytest.fixture
def tag(request, version, instance_type, processor):
    if request.config.getoption("--tag"):
        return request.config.getoption("--tag")
    return f"{version}-{processor}"


@pytest.fixture
def image_uri(registry, region, repo, tag):
    return util.image_uri(registry, region, repo, tag)


@pytest.fixture(params=os.environ["TEST_INSTANCE_TYPES"].split(","))
def instance_type(request, region):
    return request.param


@pytest.fixture(scope="module")
def accelerator_type():
    return None


@pytest.fixture(scope="session")
def tfs_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/tfs-model.tar.gz")

@pytest.fixture(scope="session")
def tfs_neuron_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/tfs-neuron-model.tar.gz")

@pytest.fixture(scope="session")
def tfs_neuronx_model(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "test/data/tfs-neuronx-model.tar.gz")


@pytest.fixture(scope="session")
def python_model_with_requirements(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/tfs-model_greater_than_equal_to_tf26.tar.gz")


@pytest.fixture(scope="session")
def python_model_with_lib(region, boto_session):
    return util.find_or_put_model_data(region,
                                       boto_session,
                                       "data/python-with-lib.tar.gz")


@pytest.fixture(scope="session")
def mme1_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, 
                                           boto_session, 
                                           "mme1", 
                                           ["test/data/mme1/code/inference.py", 
                                            "test/data/mme1/half_plus_three.tar.gz",
                                            "test/data/mme1/half_plus_two.tar.gz"])


@pytest.fixture(scope="session")
def mme2_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, 
                                           boto_session, 
                                           "mme2", 
                                           ["test/data/mme2/code/inference.py", 
                                            "test/data/mme2/half_plus_three.tar.gz",
                                            "test/data/mme2/half_plus_two.tar.gz"])


@pytest.fixture(scope="session")
def mme3_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, 
                                           boto_session, 
                                           "mme3", 
                                           ["test/data/mme3/code/inference.py", 
                                            "test/data/mme3/half_plus_three.tar.gz",
                                            "test/data/mme3/half_plus_two.tar.gz"])


@pytest.fixture(scope="session")
def mme4_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, 
                                           boto_session, 
                                           "mme4", 
                                           ["test/data/mme4/half_plus_three.tar.gz",
                                            "test/data/mme4/half_plus_two.tar.gz"])


@pytest.mark.model("unknown_model")
def test_tfs_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_model,
                                    image_uri, instance_type, accelerator_type, input_data)

@pytest.mark.model("unknown_model")
@pytest.mark.neuron_test
def test_tfs_neuron_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_neuron_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {"instances": [[[[1, 10], [2, 20]]]]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_neuron_model,
                                    image_uri, instance_type, accelerator_type, input_data)


@pytest.mark.skip("CreateEndpointConfig doesn't support trn1: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html#sagemaker-Type-ProductionVariant-InstanceType")
@pytest.mark.model("unknown_model")
@pytest.mark.neuronx_test
def test_tfs_neuronx_model(boto_session, sagemaker_client,
                   sagemaker_runtime_client, model_name, tfs_neuronx_model,
                   image_uri, instance_type, accelerator_type):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                    sagemaker_runtime_client, model_name, tfs_neuronx_model,
                                    image_uri, instance_type, accelerator_type, input_data)


@pytest.mark.integration("batch_transform")
@pytest.mark.model("unknown_model")
def test_batch_transform(region, boto_session, sagemaker_client,
                         model_name, tfs_model, image_uri,
                         instance_type):
                         
    if "graviton" in image_uri:
        pytest.skip("Test not supported with Graviton test instance.")

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
    input_data = {"instances": [[1.0, 2.0, 5.0]]}
    output_data = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                                  sagemaker_runtime_client, model_name,
                                                  python_model_with_requirements, image_uri,
                                                  instance_type, accelerator_type, input_data)
    


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


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
def test_mme1(boto_session, sagemaker_client,
              sagemaker_runtime_client, model_name, mme1_models,
              image_uri, instance_type, accelerator_type, region):
    
    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data =  {"instances": [1.0, 2.0, 5.0]}
    bucket = util._test_bucket(region, boto_session)
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": bucket,
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": "test-tfs/mme1/code/"  
        }
    outputs = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                              sagemaker_runtime_client, model_name, mme1_models,
                                              image_uri, instance_type, accelerator_type, input_data,
                                              is_multi_model_mode_enabled = True, 
                                              target_models = ["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
                                              environment = custom_env)
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
def test_mme2(boto_session, sagemaker_client,
              sagemaker_runtime_client, model_name, mme2_models,
              image_uri, instance_type, accelerator_type, region):
    
    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data =  "1.0,2.0,5.0"
    bucket = util._test_bucket(region, boto_session)
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": bucket,
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": "test-tfs/mme2/code/"  
        }
    outputs = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                              sagemaker_runtime_client, model_name, mme2_models,
                                              image_uri, instance_type, accelerator_type, input_data,
                                              is_multi_model_mode_enabled = True, 
                                              target_models = ["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
                                              environment = custom_env, content_type = "text/csv")
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
def test_mme3(boto_session, sagemaker_client,
              sagemaker_runtime_client, model_name, mme3_models,
              image_uri, instance_type, accelerator_type, region):
    
    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data =  "1.0,2.0,5.0"
    bucket = util._test_bucket(region, boto_session)
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": bucket,
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": "test-tfs/mme3/code/"  
        }
    outputs = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                              sagemaker_runtime_client, model_name, mme3_models,
                                              image_uri, instance_type, accelerator_type, input_data,
                                              is_multi_model_mode_enabled = True, 
                                              target_models = ["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
                                              environment = custom_env, content_type = "text/csv")
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
def test_mme4(boto_session, sagemaker_client,
              sagemaker_runtime_client, model_name, mme4_models,
              image_uri, instance_type, accelerator_type):
    
    if "p3" in instance_type:
        pytest.skip("skip for p3 instance")

    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data =  {"instances": [1.0, 2.0, 5.0]}
    outputs = util.create_and_invoke_endpoint(boto_session, sagemaker_client,
                                              sagemaker_runtime_client, model_name, mme4_models,
                                              image_uri, instance_type, accelerator_type, input_data,
                                              is_multi_model_mode_enabled = True, 
                                              target_models = ["half_plus_three.tar.gz", "half_plus_two.tar.gz"])
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}