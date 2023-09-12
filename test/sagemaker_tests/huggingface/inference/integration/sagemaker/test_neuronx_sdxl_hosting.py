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

import os
import pytest
import sagemaker
import boto3
from test.test_utils import (
    get_framework_and_version_from_tag,
    get_sha_of_an_image_from_ecr,
)
from sagemaker.huggingface import HuggingFaceModel


from ...integration import (
    model_dir,
    pt_model,
    script_dir,
    pt_neuronx_sdxl_script,
    dump_logs_from_cloudwatch,
)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function


# This version of the test is being added to test the neuronx inference images on multiple
# instances in the regions corresponding to their availability.
# In future, we would like to configure the logic to run multiple `pytest` commands that can allow
# us to test multiple instances in multiple regions for each image.
@pytest.mark.model("tiny-sdxl-base")
@pytest.mark.processor("neuronx")
@pytest.mark.parametrize(
    "test_region,test_instance_type",
    [("us-east-1", "ml.trn1.2xlarge"), ("us-east-2", "ml.inf2.xlarge")],
)
@pytest.mark.neuronx_test
def test_neuronx_hosting_all_instances(
    test_region, test_instance_type, instance_type, framework_version, ecr_image, py_version
):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" not in framework:
        pytest.skip(f"Skipping test for non-pytorch image - {ecr_image}")
    valid_instance_types_for_this_test = ["ml.trn1.2xlarge", "ml.inf2.xlarge"]
    assert not instance_type or instance_type in valid_instance_types_for_this_test, (
        f"Instance type value passed by pytest is {instance_type}. "
        f"This method will only test instance types in {valid_instance_types_for_this_test}"
    )
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=[test_region],
        test_function=_test_pt_neuronx,
        framework_version=framework_version,
        instance_type=test_instance_type,
        model_dir=model_dir,
        script_dir=script_dir,
        py_version=py_version,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _get_endpoint_prefix_name(custom_prefix, region_name, image_uri):
    """
    Creates an endpoint prefix name that has first 10 chars of image sha and the
    CODEBUILD_RESOLVED_SOURCE_VERSION to allow tracking of SM Endpoint Logs.

    custom_prefix: str, Initial prefix that the user wants to have in the endpoint name
    region_name: str, region_name where image is located
    image_uri: str, URI of the image
    """
    endpoint_name_prefix = custom_prefix
    try:
        image_sha = get_sha_of_an_image_from_ecr(
            ecr_client=boto3.Session(region_name=region_name).client("ecr"), image_uri=image_uri
        )
        ## Image SHA returned looks like sha256:1abc.....
        ## We extract ID from that
        image_sha_id = image_sha.split(":")[-1]
        endpoint_name_prefix = f"{endpoint_name_prefix}-{image_sha_id[:10]}"
    except:
        pass

    resolved_src_version = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "temp")
    endpoint_name_prefix = f"{endpoint_name_prefix}-{resolved_src_version}"
    return endpoint_name_prefix


def _test_pt_neuronx(
    sagemaker_session,
    framework_version,
    image_uri,
    instance_type,
    model_dir,
    script_dir,
    py_version,
    accelerator_type=None,
    **kwargs,
):
    endpoint_name_prefix = _get_endpoint_prefix_name(
        custom_prefix="sm-hf-neuronx-sdxl-serving",
        region_name=sagemaker_session.boto_region_name,
        image_uri=image_uri,
    )
    endpoint_name = sagemaker.utils.unique_name_from_base(endpoint_name_prefix)

    # dummy pytorch model, not used for the test
    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-neuronx-serving/models",
    )
    model_file = pt_model

    # test script using `optimum-neuron`
    entry_point = pt_neuronx_sdxl_script

    hf_model = HuggingFaceModel(
        model_data=f"{model_data}/{model_file}",
        role="SageMakerRole",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
        entry_point=entry_point,
        source_dir=script_dir,
        py_version=py_version,
        model_server_workers=1,
        env={"AWS_NEURON_VISIBLE_DEVICES": "ALL"},
    )
    hf_model._is_compiled_model = True

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        data = {"inputs": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"}
        output = predictor.predict(data)

        assert "generated_image" in output
