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
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import BytesDeserializer

from ...integration import model_neuron_dir, resnet_neuron_script, resnet_neuron_input, resnet_neuron_image_list
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function


@pytest.mark.model("resnet")
@pytest.mark.processor("neuron")
@pytest.mark.neuron_test
def test_neuron_hosting(framework_version, ecr_image, instance_type, sagemaker_regions):
    import pdb; pdb.set_trace()
    instance_type = instance_type or 'ml.inf1.xlarge'
    model_dir = os.path.join(model_neuron_dir, 'model-resnet.tar.gz')
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'model_dir': model_dir,
            'resnet_script': resnet_neuron_script,
            'resnet_neuron_input': resnet_neuron_input,
            'resnet_neuron_image_list': resnet_neuron_image_list,

        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_resnet_distributed, function_args)


def _test_resnet_distributed(
        ecr_image, sagemaker_session, framework_version, instance_type, model_dir, resnet_script, resnet_neuron_input, resnet_neuron_image_list, accelerator_type=None
):
    import pdb; pdb.set_trace()
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-pytorch-serving")

    model_data = sagemaker_session.upload_data(
        path=model_dir,
        key_prefix="sagemaker-pytorch-serving/models",
    )

    pytorch = PyTorchModel(
        model_data=model_data,
        role='SageMakerRole',
        entry_point=resnet_script,
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        model_server_workers=4,
        env={"AWS_NEURON_VISIBLE_DEVICES": "ALL", "NEURONCORE_GROUP_SIZES":"1", "NEURON_RT_VISIBLE_CORES": "0", "NEURON_RT_LOG_LEVEL":"5", "NEURON_RTD_ADDRESS":"run"}
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = pytorch.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=IdentitySerializer(), 
            deserializer=BytesDeserializer(),
        )

        with open(resnet_neuron_input, "rb") as f:
            payload = f.read()
        output = predictor.predict(data=payload)
        print(output)
        result = json.loads(output.decode())
        print(result)

        # Load names for ImageNet classes
        object_categories = {}
        with open(resnet_neuron_image_list, "r") as f:
            for line in f:
                key, val = line.strip().split(":")
                object_categories[key] = val
        
        assert("cat" in object_categories[str(np.argmax(result))])