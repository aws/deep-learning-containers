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
    dump_logs_from_cloudwatch,
    model_dir,
    pt_model,
    pt_diffusers_cpu_script,
    pt_diffusers_gpu_script,
    script_dir,
)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function


@pytest.mark.model("tiny-stable-diffusion")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_diffusers_cpu_hosting(
    framework_version, ecr_image, instance_type, sagemaker_regions, py_version
):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" not in framework:
        pytest.skip(f"Skipping test for non-pytorch image - {ecr_image}")
    instance_type = instance_type or "ml.m5.xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_diffusion_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        script_dir=script_dir,
        py_version=py_version,
        processor="cpu",
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


# Only test normal size model for gpu as cpu time out
@pytest.mark.model("stable-diffusion")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_diffusers_gpu_hosting(
    framework_version, ecr_image, instance_type, sagemaker_regions, py_version
):
    framework, _ = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" not in framework:
        pytest.skip(f"Skipping test for non-pytorch image - {ecr_image}")
    instance_type = instance_type or "ml.p3.2xlarge"
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_diffusion_model,
        framework_version=framework_version,
        instance_type=instance_type,
        model_dir=model_dir,
        script_dir=script_dir,
        py_version=py_version,
        processor="gpu",
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _test_diffusion_model(
    sagemaker_session,
    framework_version,
    image_uri,
    instance_type,
    model_dir,
    script_dir,
    py_version,
    processor,
    **kwargs,
):
    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-huggingface-serving-diffusion-model-serving"
    )

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-inference-diffusers-serving/models",
    )
    entry_script = {
        "cpu": pt_diffusers_cpu_script,
        "gpu": pt_diffusers_gpu_script,
    }

    model_file = pt_model
    entry_point = entry_script[processor]

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

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        num_images_per_prompt = 1

        prompt = (
            "A dog trying catch a flying pizza art drawn by disney concept artists, golden colour,"
            " high quality, highly detailed, elegant, sharp focus"
        )
        output = predictor.predict(
            data={"inputs": prompt, "num_images_per_prompt": num_images_per_prompt}
        )

        assert "generated_images" in output
