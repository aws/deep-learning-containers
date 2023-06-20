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

import pytest
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

from test.test_utils import get_framework_and_version_from_tag
from ...integration import (
    model_dir,
    pt_model,
    script_dir,
    pt_ipex_script,
    dump_logs_from_cloudwatch,
)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function


@pytest.mark.model("tiny-distilbert")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_ipex_hosting(framework_version, ecr_image, instance_type, sagemaker_regions, py_version):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" not in framework:
        pytest.skip(f"Skipping test for non-pytorch image - {ecr_image}")
    instance_type = instance_type or "ml.m5.xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_pt_ipex,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        script_dir=script_dir,
        py_version=py_version,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _test_pt_ipex(
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
    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-huggingface-inference-ipex-serving"
    )

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-inference-ipex-serving/models",
    )

    model_file = pt_model
    entry_point = pt_ipex_script

    hf_model = HuggingFaceModel(
        model_data=f"{model_data}/{model_file}",
        role="SageMakerRole",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
        entry_point=entry_point,
        source_dir=script_dir,
        py_version=py_version,
        model_server_workers=1,
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
