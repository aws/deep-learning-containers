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
from __future__ import absolute_import

import os
import pytest
import sagemaker
from sagemaker import utils
from sagemaker.pytorch import PyTorch
from ...integration import (neuron_allreduce_path, DEFAULT_TIMEOUT)
from ...integration.sagemaker.timeout import timeout
from .... import invoke_pytorch_helper_function

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


@pytest.mark.processor("neuron")
@pytest.mark.model("unknown_model")
@pytest.mark.neuron_test
def test_neuron_allreduce_single_node(framework_version, ecr_image, sagemaker_regions, instance_type):
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'num_neuron_cores': 2,
        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_neuron_allreduce, function_args)

def _test_neuron_allreduce(
        ecr_image, sagemaker_session, framework_version, instance_type, instance_count=1, num_neuron_cores=2
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point='entrypoint.py',
            source_dir=neuron_allreduce_path,
            role='SageMakerRole',
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={'nproc-per-node': num_neuron_cores, 'nnodes': instance_count},
            disable_profiler=True,
            env={"NEURON_RT_LOG_LEVEL": "DEBUG"}
        )

        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=neuron_allreduce_path, key_prefix='pytorch/distributed_operations'
        )

        pytorch.fit({'required_argument': fake_input}, job_name=utils.unique_name_from_base('test-pt-neuron-allreduce'))
