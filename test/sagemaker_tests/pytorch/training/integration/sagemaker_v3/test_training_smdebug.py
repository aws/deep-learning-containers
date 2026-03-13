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

import os

import pytest
from sagemaker.train.configs import SourceCode

from ...integration import training_dir, smdebug_mnist_script, DEFAULT_TIMEOUT
from .timeout import timeout
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer


@pytest.mark.skip("SM Debugger/Profiler v1 deprecated")
@pytest.mark.skip_py2_containers
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.team("smdebug")
def test_training_smdebug(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)

    hyperparameters = {
        "random_seed": True,
        "num_steps": 50,
        "smdebug_path": "/tmp/ml/output/tensors",
        "epochs": 1,
        "data_dir": training_dir,
    }

    source_code = SourceCode(
        source_dir=os.path.dirname(smdebug_mnist_script),
        entry_script=os.path.basename(smdebug_mnist_script),
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-v3-smdebug-training",
        )


@pytest.mark.skip("SM Debugger/Profiler v1 deprecated")
@pytest.mark.skip_py2_containers
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.integration("smdebug")
@pytest.mark.model("mnist")
@pytest.mark.team("smdebug")
def test_hc_training_smdebug(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)

    hyperparameters = {
        "random_seed": True,
        "num_steps": 50,
        "smdebug_path": "/tmp/ml/output/tensors",
        "epochs": 1,
        "data_dir": training_dir,
    }

    source_code = SourceCode(
        source_dir=os.path.dirname(smdebug_mnist_script),
        entry_script=os.path.basename(smdebug_mnist_script),
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-v3-hc-smdebug-training",
        )
