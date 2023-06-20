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
from ...... import invoke_sm_endpoint_helper_function


MODEL_WITH_REQUIREMENTS_PATH = "data/tfs-model_greater_than_equal_to_tf26.tar.gz"
MODEL_WITH_LIB_PATH = "data/python-with-lib.tar.gz"
TFS_MODEL_PATH = "data/tfs-model.tar.gz"
TFS_NEURONX_MODEL_PATH = "data/tfs-neuronx-model.tar.gz"
TFS_NEURON_MODEL_PATH = "data/tfs-neuron-model.tar.gz"
MME1_MODEL_PATHS = [
    "test/data/mme1/code/inference.py",
    "test/data/mme1/half_plus_three.tar.gz",
    "test/data/mme1/half_plus_two.tar.gz",
]
MME2_MODEL_PATHS = [
    "test/data/mme2/code/inference.py",
    "test/data/mme2/half_plus_three.tar.gz",
    "test/data/mme2/half_plus_two.tar.gz",
]
MME3_MODEL_PATHS = [
    "test/data/mme3/code/inference.py",
    "test/data/mme3/half_plus_three.tar.gz",
    "test/data/mme3/half_plus_two.tar.gz",
]
MME4_MODEL_PATHS = ["test/data/mme4/half_plus_three.tar.gz", "test/data/mme4/half_plus_two.tar.gz"]


@pytest.fixture(params=os.environ["TEST_VERSIONS"].split(","))
def version(request):
    return request.param


@pytest.fixture(scope="session")
def repo(request):
    return request.config.getoption("--repo") or "sagemaker-tensorflow-serving"


@pytest.fixture
def processor(request, instance_type):
    return request.config.getoption("--processor") or (
        "gpu" if instance_type.startswith("ml.p") or instance_type.startswith("ml.g") else "cpu"
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
    return util.find_or_put_model_data(region, boto_session, TFS_MODEL_PATH)


@pytest.fixture(scope="session")
def python_model_with_requirements(region, boto_session):
    return util.find_or_put_model_data(region, boto_session, MODEL_WITH_REQUIREMENTS_PATH)


@pytest.fixture(scope="session")
def python_model_with_lib(region, boto_session):
    return util.find_or_put_model_data(region, boto_session, MODEL_WITH_LIB_PATH)


@pytest.fixture(scope="session")
def mme1_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, boto_session, "mme1", MME1_MODEL_PATHS)


@pytest.fixture(scope="session")
def mme2_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, boto_session, "mme2", MME2_MODEL_PATHS)


@pytest.fixture(scope="session")
def mme3_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, boto_session, "mme3", MME3_MODEL_PATHS)


@pytest.fixture(scope="session")
def mme4_models(region, boto_session):
    return util.find_or_put_mme_model_data(region, boto_session, "mme4", MME4_MODEL_PATHS)


@pytest.mark.model("unknown_model")
def test_tfs_model(sagemaker_regions, model_name, image_uri, instance_type, accelerator_type):
    input_data = {"instances": [1.0, 2.0, 5.0]}
    invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.create_and_invoke_endpoint,
        local_model_paths=[TFS_MODEL_PATH],
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
    )


@pytest.mark.model("unknown_model")
@pytest.mark.neuron_test
def test_tfs_neuron_model(
    model_name, sagemaker_regions, image_uri, instance_type, accelerator_type
):
    input_data = {"instances": [[[[1, 10], [2, 20]]]]}
    invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.create_and_invoke_endpoint,
        local_model_paths=[TFS_NEURON_MODEL_PATH],
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
    )


@pytest.mark.model("unknown_model")
@pytest.mark.neuronx_test
def test_tfs_neuronx_model(
    model_name, sagemaker_regions, image_uri, instance_type, accelerator_type
):
    input_data = {"instances": [[1.0, 2.0, 5.0]]}
    invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.create_and_invoke_endpoint,
        local_model_paths=[TFS_NEURONX_MODEL_PATH],
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
    )


@pytest.mark.integration("batch_transform")
@pytest.mark.model("unknown_model")
def test_batch_transform(sagemaker_regions, model_name, image_uri, instance_type):
    if "graviton" in image_uri:
        pytest.skip("Test not supported with Graviton test instance.")

    results = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.run_batch_transform_job,
        local_model_paths=[TFS_MODEL_PATH],
        model_name=model_name,
        instance_type=instance_type,
    )
    assert len(results) == 10
    for r in results:
        assert r == [3.5, 4.0, 5.5]


@pytest.mark.model("unknown_model")
def test_python_model_with_requirements(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    # the python service needs to transform this to get a valid prediction
    input_data = {"instances": [[1.0, 2.0, 5.0]]}
    invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.create_and_invoke_endpoint,
        local_model_paths=[MODEL_WITH_REQUIREMENTS_PATH],
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
    )


@pytest.mark.model("unknown_model")
def test_python_model_with_lib(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    # the python service needs to transform this to get a valid prediction
    input_data = {"x": [1.0, 2.0, 5.0]}
    output_data = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_model_data,
        test_function=util.create_and_invoke_endpoint,
        local_model_paths=[MODEL_WITH_LIB_PATH],
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
    )

    # python service adds this to tfs response
    assert output_data["python"] is True
    assert output_data["dummy_module"] == "0.1"


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
def test_mme1(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data = {"instances": [1.0, 2.0, 5.0]}
    mme_folder_name = "mme1"
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": "placeholder_bucket",
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": f"test-tfs/{mme_folder_name}/code/",
        "SAGEMAKER_GUNICORN_WORKERS": "5",
    }
    outputs = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_mme_model_data,
        test_function=_mme_test_helper,
        mme_folder_name=mme_folder_name,
        local_model_paths=MME1_MODEL_PATHS,
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
        is_multi_model_mode_enabled=True,
        target_models=["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
        environment=custom_env,
    )
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
def test_mme2(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data = "1.0,2.0,5.0"
    mme_folder_name = "mme2"
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": "placeholder_bucket",
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": f"test-tfs/{mme_folder_name}/code/",
        "SAGEMAKER_GUNICORN_WORKERS": "5",
    }
    outputs = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_mme_model_data,
        test_function=_mme_test_helper,
        mme_folder_name=mme_folder_name,
        local_model_paths=MME2_MODEL_PATHS,
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
        is_multi_model_mode_enabled=True,
        target_models=["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
        environment=custom_env,
        content_type="text/csv",
    )
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
def test_mme3(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data = "1.0,2.0,5.0"
    mme_folder_name = "mme3"
    custom_env = {
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET": "placeholder_bucket",
        "SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX": f"test-tfs/{mme_folder_name}/code/",
        "SAGEMAKER_GUNICORN_WORKERS": "5",
    }
    outputs = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_mme_model_data,
        test_function=_mme_test_helper,
        mme_folder_name=mme_folder_name,
        local_model_paths=MME3_MODEL_PATHS,
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
        is_multi_model_mode_enabled=True,
        target_models=["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
        environment=custom_env,
        content_type="text/csv",
    )
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


@pytest.mark.integration("mme")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
def test_mme4(
    sagemaker_regions,
    model_name,
    image_uri,
    instance_type,
    accelerator_type,
):
    if "graviton" in image_uri:
        pytest.skip("MME test not supported with Graviton test instance.")

    # the python service needs to transform this to get a valid prediction
    input_data = {"instances": [1.0, 2.0, 5.0]}
    mme_folder_name = "mme4"
    outputs = invoke_sm_endpoint_helper_function(
        ecr_image=image_uri,
        sagemaker_regions=sagemaker_regions,
        model_helper=util.find_or_put_mme_model_data,
        test_function=_mme_test_helper,
        mme_folder_name=mme_folder_name,
        local_model_paths=MME4_MODEL_PATHS,
        model_name=model_name,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        input_data=input_data,
        is_multi_model_mode_enabled=True,
        target_models=["half_plus_three.tar.gz", "half_plus_two.tar.gz"],
    )
    assert outputs[0] == {"predictions": [3.5, 4.0, 5.5]}
    assert outputs[1] == {"predictions": [2.5, 3.0, 4.5]}


def _mme_test_helper(
    boto_session,
    sagemaker_client,
    sagemaker_runtime_client,
    model_name,
    model_data,
    image_uri,
    instance_type,
    accelerator_type,
    region,
    input_data,
    target_models,
    environment=None,
    is_multi_model_mode_enabled=True,
    content_type="application/json",
    **kwargs,
):
    if not environment:
        environment = {}
    bucket = util._test_bucket(region, boto_session)
    if environment.get("SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET"):
        environment["SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET"] = bucket
    outputs = util.create_and_invoke_endpoint(
        boto_session,
        sagemaker_client,
        sagemaker_runtime_client,
        model_name,
        model_data,
        image_uri,
        instance_type,
        accelerator_type,
        input_data,
        region=region,
        is_multi_model_mode_enabled=is_multi_model_mode_enabled,
        target_models=target_models,
        environment=environment,
        content_type=content_type,
    )
    return outputs
