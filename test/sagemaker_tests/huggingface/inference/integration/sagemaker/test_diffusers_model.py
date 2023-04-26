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
from sagemaker.huggingface import HuggingFaceModel
from distutils.dir_util import copy_tree
from pathlib import Path
from huggingface_hub import snapshot_download
import random
import tarfile


from ...integration import script_dir, pt_diffusers_script, dump_logs_from_cloudwatch
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint


@pytest.mark.model("stable-diffusion")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_diffusers_cpu(sagemaker_session, framework_version, ecr_image, instance_type, region):
    instance_type = instance_type or "ml.m5.xlarge"
    try:
        _test_diffusion_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir, script_dir, py_version)
    except Exception as e:
        dump_logs_from_cloudwatch(e, region)
        raise


@pytest.mark.model("stable-diffusion")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_diffusers_gpu(sagemaker_session, framework_version, ecr_image, instance_type, region):
    instance_type = instance_type or "ml.p3.2xlarge"
    try:
        _test_diffusion_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir)
    except Exception as e:
        dump_logs_from_cloudwatch(e, region)
        raise


# helper to create the model.tar.gz
def _compress(tar_dir=None, output_file="model.tar.gz"):
    parent_dir=os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir('.'):
          print(item)
          tar.add(item, arcname=item)    
    os.chdir(parent_dir)


def _test_diffusion_model(sagemaker_session, framework_version, ecr_image, instance_type, model_dir, script_dir, accelerator_type=None):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-huggingface-serving-diffusion-model")
    
    # download snapshot
    HF_MODEL_ID="CompVis/stable-diffusion-v1-4"
    snapshot_dir = snapshot_download(repo_id=HF_MODEL_ID,revision="fp16")

    # create model dir
    model_tar = Path(f"model-{random.getrandbits(16)}")
    model_tar.mkdir(exist_ok=True)

    # copy snapshot to model dir
    copy_tree(snapshot_dir, str(model_tar))

    # compress the model
    _compress(str(model_tar))
    model_dir="model.tar.gz"

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-huggingface-serving-diffusion-model/models",
    )

    if "pytorch" in ecr_image:
        entry_point = pt_diffusers_script
    else:
        raise ValueError(f"Unsupported framework for image: {ecr_image}")

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
    hf_model._is_compiled_model = True

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        num_images_per_prompt = 1

        prompt ="A dog trying catch a flying pizza art drawn by disney concept artists, golden colour, high quality, highly detailed, elegant, sharp focus"
        output = predictor.predict(
            data={
                "inputs": prompt,
                "num_images_per_prompt" : num_images_per_prompt
            }
        )

        assert "generated_images" in output