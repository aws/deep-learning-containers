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
    ecr as ecr_utils,
    get_repository_and_tag_from_image_uri,
)
from sagemaker.huggingface import HuggingFaceModel


from ...integration import (
    model_dir,
    pt_neuronx_model,
    script_dir,
    pt_neuronx_script,
    dump_logs_from_cloudwatch,
)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
def test_neuron_hosting(
    sagemaker_session, framework_version, ecr_image, instance_type, region, py_version
):
    instance_type = instance_type or "ml.inf2.xlarge"
    try:
        _test_pt_neuronx(
            sagemaker_session,
            framework_version,
            ecr_image,
            instance_type,
            model_dir,
            script_dir,
            py_version,
        )
    except Exception as e:
        dump_logs_from_cloudwatch(e, region)
        raise


## This version of the test is being added to test the neuronx inference images on multiple instances in the regions corresponding to their availability.
## In future, we would like to configure the logic to run multiple `pytest` commands that can allow us to test multiple instances in multiple regions for each image.
@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("neuronx")
@pytest.mark.parametrize(
    "test_region,test_instance_type",
    [("us-east-1", "ml.trn1.2xlarge"), ("us-east-2", "ml.inf2.xlarge")],
)
@pytest.mark.neuronx_test
def test_neuronx_hosting_trn(
    test_region, test_instance_type, region, instance_type, framework_version, ecr_image, py_version
):
    valid_instance_types_for_this_test = ["ml.trn1.2xlarge", "ml.inf2.xlarge"]
    assert (
        not instance_type or instance_type in valid_instance_types_for_this_test
    ), f"Instance type value passed by pytest is {instance_type}. This method will only test instance types in {valid_instance_types_for_this_test}"
    test_sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=test_region))
    image_repo_name, _ = get_repository_and_tag_from_image_uri(ecr_image)
    test_image_uri = ecr_utils.reupload_image_to_test_ecr(
        ecr_image, target_image_repo_name=image_repo_name, target_region=test_region
    )
    try:
        _test_pt_neuronx(
            test_sagemaker_session,
            framework_version,
            test_image_uri,
            test_instance_type,
            model_dir,
            script_dir,
            py_version,
        )
    except Exception as e:
        dump_logs_from_cloudwatch(e, test_region)
        raise


def _test_pt_neuronx(
    sagemaker_session,
    framework_version,
    ecr_image,
    instance_type,
    model_dir,
    script_dir,
    py_version,
    accelerator_type=None,
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-neuronx-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-neuronx-serving/models",
    )

    if "pytorch" in ecr_image:
        model_file = pt_neuronx_model
        entry_point = pt_neuronx_script
    else:
        raise ValueError(f"Unsupported framework for image: {ecr_image}")

    hf_model = HuggingFaceModel(
        model_data=f"{model_data}/{model_file}",
        role="SageMakerRole",
        image_uri=ecr_image,
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

        data = {
            "inputs": "Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days."
        }
        output = predictor.predict(data)

        assert "score" in output[0]
