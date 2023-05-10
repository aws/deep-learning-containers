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
from distutils.dir_util import copy_tree
from huggingface_hub import snapshot_download

from ...integration import dump_logs_from_cloudwatch, model_dir, pt_diffusers_script, script_dir
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


# helper to compress diffusion model
def _compress(tar_dir=None, output_file="model.tar.gz"):
    parent_dir = os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir("."):
            print(item)
            tar.add(item, arcname=item)
    os.chdir(parent_dir)


def _create_model(model_id, script_path, sagemaker_session):
    snapshot_dir = snapshot_download(repo_id=model_id, revision="fp16")
    model_tar = Path(f"model-{random.getrandbits(16)}")
    model_tar.mkdir(exist_ok=True)
    copy_tree(snapshot_dir, str(model_tar))

    os.makedirs(str(model_tar.joinpath("code")), exist_ok=True)
    shutil.copy(script_path, str(model_tar.joinpath("code/inference.py")))
    _compress(str(model_tar))

    model_data = sagemaker_session.upload_data(
        path="model.tar.gz",
        key_prefix="sagemaker-huggingface-serving-diffusion-model-serving",
    )

    return model_data


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
    
    HF_MODEL_ID = "CompVis/stable-diffusion-v1-4"

    model_data = _create_model(
        model_id=HF_MODEL_ID,
        script_path=os.path.join(script_dir, entry_point),
        sagemaker_session=sagemaker_session,
    )
    print(f"model data: {model_data}")


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
