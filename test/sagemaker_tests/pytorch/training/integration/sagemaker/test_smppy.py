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
import time

import boto3
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import Torchrun

from test.test_utils import get_framework_and_version_from_tag
from ...integration import DEFAULT_TIMEOUT, smppy_mnist_script, training_dir
from ...integration.sagemaker.timeout import timeout
from . import invoke_pytorch_training
from .test_torch_distributed import validate_or_skip_distributed_training

INSTANCE_TYPE = "ml.g4dn.12xlarge"


def _skip_if_image_is_not_compatible_with_smppy(image_uri):
    _, framework_version = get_framework_and_version_from_tag(image_uri)
    compatible_versions = SpecifierSet(">=2.0")
    if Version(framework_version) not in compatible_versions:
        pytest.skip(f"This test only works for PT versions in {compatible_versions}")


@pytest.mark.skip_smppy_test
@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy(framework_version, ecr_image, sagemaker_regions):
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):
        source_code = SourceCode(
            entry_script=smppy_mnist_script,
        )
        
        compute = Compute(
            instance_type=INSTANCE_TYPE,
            instance_count=1,
        )
        
        hyperparameters = {"epochs": 1}

        model_trainer, _ = invoke_pytorch_training(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute=compute,
            hyperparameters=hyperparameters,
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-smppy-training",
        )
        # Note: Profiler config is handled differently in v3
        # The profiler functionality may need separate configuration


@pytest.mark.skip_smppy_test
@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy_distributed(framework_version, ecr_image, sagemaker_regions):
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_distributed_training(ecr_image)
        
        source_code = SourceCode(
            entry_script=smppy_mnist_script,
        )
        
        compute = Compute(
            instance_type=INSTANCE_TYPE,
            instance_count=2,
        )
        
        hyperparameters = {"epochs": 1}

        model_trainer, _ = invoke_pytorch_training(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute=compute,
            hyperparameters=hyperparameters,
            distributed_runner=Torchrun(),
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-smppy-training-distributed",
        )
        # Note: Profiler config is handled differently in v3
        # The profiler functionality may need separate configuration
