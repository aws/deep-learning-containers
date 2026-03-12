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
from sagemaker.modules.configs import SourceCode

from test.test_utils import get_framework_and_version_from_tag
from ...integration import DEFAULT_TIMEOUT, smppy_mnist_script, training_dir, mnist_path
from .timeout import timeout
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer
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
    skip_if_not_v3_compatible(ecr_image)
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)

    source_code = SourceCode(
        source_dir=os.path.dirname(smppy_mnist_script),
        entry_script=os.path.basename(smppy_mnist_script),
    )
    compute_params = {"instance_type": INSTANCE_TYPE, "instance_count": 1}
    hyperparameters = {"epochs": 1}

    # TODO: ProfilerConfig/Profiler from SM SDK v2 does not have a direct v3 equivalent yet.
    # Profiling configuration is omitted for now. Add v3 profiling support when available.

    with timeout(minutes=DEFAULT_TIMEOUT):
        model_trainer, _ = invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-v3-smppy-training",
        )


@pytest.mark.skip_smppy_test
@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy_distributed(framework_version, ecr_image, sagemaker_regions):
    skip_if_not_v3_compatible(ecr_image)
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)
    validate_or_skip_distributed_training(ecr_image)

    from sagemaker.modules.distributed import Torchrun

    source_code = SourceCode(
        source_dir=os.path.dirname(smppy_mnist_script),
        entry_script=os.path.basename(smppy_mnist_script),
    )
    compute_params = {"instance_type": INSTANCE_TYPE, "instance_count": 2}
    hyperparameters = {"epochs": 1}
    distributed_runner = Torchrun()

    # TODO: ProfilerConfig/Profiler from SM SDK v2 does not have a direct v3 equivalent yet.
    # Profiling configuration is omitted for now. Add v3 profiling support when available.

    with timeout(minutes=DEFAULT_TIMEOUT):
        model_trainer, _ = invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            distributed_runner=distributed_runner,
            upload_s3_data_args={"path": training_dir, "key_prefix": "pytorch/mnist"},
            job_name="test-pt-v3-smppy-training-distributed",
        )
