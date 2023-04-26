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
import random
import shutil
import tarfile
from pathlib import Path

import pytest
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

from ...integration import (dump_logs_from_cloudwatch, model_dir,
                            pt_diffusers_script, script_dir)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.model("stable-diffusion")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_diffusers_gpu(
    sagemaker_session, framework_version, ecr_image, instance_type, region, py_version
):
    instance_type = instance_type or "ml.g4dn.xlarge"
    try:
        _test_diffusion_model(
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


# helper to create a asset.tar.gz
def _compress(file_path, zip_file_path="asset.tar.gz"):
    asset_dest = Path(f"asset-{random.getrandbits(41)}/code")
    asset_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_path, asset_dest)
    with tarfile.open(zip_file_path, "w:gz") as tar:
        for item in os.listdir(asset_dest.parent):
            tar.add(item, arcname=item)
            print(item)
    return zip_file_path


def _test_diffusion_model(
    sagemaker_session,
    framework_version,
    ecr_image,
    instance_type,
    model_dir,
    script_dir,
    py_version,
    accelerator_type=None,
):

    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-huggingface-serving-diffusion-model-serving"
    )

    if "pytorch" in ecr_image:
        entry_point = pt_diffusers_script
    else:
        raise ValueError(f"Unsupported framework for image: {ecr_image}")

    compressed_path = _compress(os.path.join(script_dir, entry_point))

    model_data = sagemaker_session.upload_data(
        path=compressed_path,
        key_prefix="sagemaker-huggingface-serving-diffusion-model-serving/asset",
    )

    hf_model = HuggingFaceModel(
        model_data=model_data,
        role="SageMakerRole",
        image_uri=ecr_image,
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

        prompt = "A dog trying catch a flying pizza art drawn by disney concept artists, golden colour, high quality, highly detailed, elegant, sharp focus"
        output = predictor.predict(
            data={"inputs": prompt, "num_images_per_prompt": num_images_per_prompt}
        )

        assert "generated_images" in output
