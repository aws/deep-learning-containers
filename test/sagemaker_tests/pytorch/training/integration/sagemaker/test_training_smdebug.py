# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import pytest
from sagemaker.pytorch import PyTorch

from ...integration import training_dir, smdebug_mnist_script, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout


@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
def test_training_smdebug(sagemaker_session, ecr_image, instance_type):
    hyperparameters = {'random_seed': True, 'num_steps': 50, 'smdebug_path': '/tmp/ml/output/tensors', 'epochs': 1,
                       'data_dir': training_dir}

    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(entry_point=smdebug_mnist_script,
                          role='SageMakerRole',
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session,
                          image_name=ecr_image,
                          hyperparameters=hyperparameters)
        training_input = pytorch.sagemaker_session.upload_data(path=training_dir,
                                                               key_prefix='pytorch/mnist')
        pytorch.fit({'training': training_input})
