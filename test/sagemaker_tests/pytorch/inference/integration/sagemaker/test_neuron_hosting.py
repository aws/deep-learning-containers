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

import numpy as np
import json
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from sagemaker.deserializers import BytesDeserializer

from ...integration import (
    model_neuron_dir,
    resnet_neuron_script,
    resnet_neuron_input,
    resnet_neuron_image_list,
)
from ...integration import (
    model_neuronx_dir,
    resnet_neuronx_script,
    resnet_neuronx_input,
    resnet_neuronx_image_list,
)
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function


@pytest.mark.model("resnet")
@pytest.mark.processor("neuron")
@pytest.mark.neuron_test
@pytest.mark.skip_stabilityai
def test_neuron_hosting(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = instance_type or "ml.inf1.xlarge"
    model_dir = os.path.join(model_neuron_dir, "model-resnet.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
        "resnet_script": resnet_neuron_script,
        "resnet_neuron_input": resnet_neuron_input,
        "resnet_neuron_image_list": resnet_neuron_image_list,
        "accelerator_type": "neuron",
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_resnet_distributed, function_args
    )


# parametrize is temporary until trn1 and inf2 become available in the same region
@pytest.mark.model("resnet")
@pytest.mark.processor("neuronx")
@pytest.mark.parametrize(
    "instance_type, sagemaker_regions",
    [("ml.trn1.2xlarge", ["us-east-1"]), ("ml.inf2.xlarge", ["us-east-2"])],
)
@pytest.mark.neuronx_test
@pytest.mark.skip_stabilityai
def test_neuronx_hosting(framework_version, ecr_image, instance_type, sagemaker_regions):
    if "neuronx" not in ecr_image:
        pytest.skip("Not a NeuronX DLC")
    model_dir = os.path.join(model_neuronx_dir, "model-resnet.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
        "resnet_script": resnet_neuronx_script,
        "resnet_neuron_input": resnet_neuronx_input,
        "resnet_neuron_image_list": resnet_neuronx_image_list,
        "accelerator_type": "neuronx",
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_resnet_distributed, function_args
    )


# parametrize is temporary until trn1 and inf2 become available in the same region
@pytest.mark.model("resnet")
@pytest.mark.processor("neuronx")
@pytest.mark.parametrize(
    "instance_type, sagemaker_regions",
    [("ml.trn1.2xlarge", ["us-east-1"]), ("ml.inf2.xlarge", ["us-east-2"])],
)
@pytest.mark.neuronx_test
@pytest.mark.skip_stabilityai
def test_neuronx_hosting_no_script(framework_version, ecr_image, instance_type, sagemaker_regions):
    if "neuronx" not in ecr_image:
        pytest.skip("Not a NeuronX DLC")
    model_dir = os.path.join(model_neuronx_dir, "model-resnet.tar.gz")
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_dir": model_dir,
        "resnet_script": None,
        "resnet_neuron_input": resnet_neuronx_input,
        "resnet_neuron_image_list": resnet_neuronx_image_list,
        "preprocess_image": True,
        "accelerator_type": "neuronx",
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_resnet_distributed, function_args
    )


def _test_resnet_distributed(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    model_dir,
    resnet_script,
    resnet_neuron_input,
    resnet_neuron_image_list,
    preprocess_image=False,
    accelerator_type=None,
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        entry_point=resnet_script,
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        model_server_workers=2,
        env={
            "AWS_NEURON_VISIBLE_DEVICES": "ALL",
            "NEURON_RT_NUM_CORES": "1",
            "NEURON_RT_LOG_LEVEL": "5",
        },
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        serializer = IdentitySerializer()
        if preprocess_image:
            serializer = JSONSerializer()
        predictor = pytorch.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=serializer,
            deserializer=BytesDeserializer(),
        )

        with open(resnet_neuron_input, "rb") as f:
            payload = f.read()

        if preprocess_image:
            import io
            import torchvision.transforms as transforms
            from PIL import Image

            data = io.BytesIO(payload)
            input_image = Image.open(data).convert("RGB")
            preprocess = transforms.Compose(
                [
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            payload = input_batch.tolist()

        output = predictor.predict(data=payload)
        print(output)
        result = json.loads(output.decode())
        print(result)

        # Load names for ImageNet classes
        object_categories = {}
        if accelerator_type == "neuronx":
            with open(resnet_neuron_image_list, "r") as f:
                object_categories = json.load(f)
            assert "cat" in object_categories[str(np.argmax(result))][1]
            return
        with open(resnet_neuron_image_list, "r") as f:
            for line in f:
                key, val = line.strip().split(":")
                object_categories[key] = val

        assert "cat" in object_categories[str(np.argmax(result))]
