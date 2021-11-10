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

import pytest
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig

from ...integration import ROLE, distrilbert_script
import unittest.mock as mock


@pytest.mark.model("hf_bert")
@pytest.mark.integration("sagemaker-training-compiler")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_huggingface_containers
@pytest.mark.skip(reason="WIP: Currently hangs")
@mock.patch('sagemaker.huggingface.TrainingCompilerConfig.validate', return_value=None)
def test_single_node_single_gpu_tcc_default(patched, docker_image, processor, instance_type, sagemaker_local_session, py_version, capsys):
    '''
    Single GPU test that tests the local_gpu instance type with default TCC.
    All local mode tests (PT and TF) are run serially on a single instance.
    '''
    hyperparameters = {"max_steps": 3, "train_batch_size": 4, "model_name": "distilbert-base-uncased"}

    estimator = HuggingFace(
        compiler_config=TrainingCompilerConfig(),
        entry_point=distrilbert_script,
        instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        image_uri=docker_image,
        instance_count=1,
        role=ROLE,
        hyperparameters=hyperparameters,
        environment={'GPU_NUM_DEVICES':'1'}, #https://github.com/aws/sagemaker-training-toolkit/issues/107
        py_version=py_version,
    )

    estimator.fit()

