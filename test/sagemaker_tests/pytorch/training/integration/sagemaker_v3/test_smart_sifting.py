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
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from sagemaker.train.configs import SourceCode

from .timeout import timeout
from ...integration import smart_sifting_path, DEFAULT_TIMEOUT
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag


def validate_or_skip_smart_sifting(ecr_image):
    if not can_run_smart_sifting(ecr_image):
        pytest.skip("Smart sifting is only available for use with PT 2.0.1")


def can_run_smart_sifting(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(image_uri=ecr_image)
    return Version(image_framework_version) in SpecifierSet("==2.0.*") and (
        not image_cuda_version or image_cuda_version == "cu118"
    )


@pytest.mark.usefixtures("feature_smart_sifting_present")
@pytest.mark.processor("cpu")
@pytest.mark.model("bert")
@pytest.mark.integration("smart_sifting")
@pytest.mark.skip_gpu
@pytest.mark.skip_py2_containers
def test_smart_sifting_cpu(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smart_sifting(ecr_image)
    instance_type = instance_type or "ml.c5.xlarge"

    source_code = SourceCode(
        source_dir=smart_sifting_path,
        entry_script="train_plt_smart_sifting.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}
    hyperparameters = {"epochs": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-smart-sifting",
        )


@pytest.mark.usefixtures("feature_smart_sifting_present")
@pytest.mark.processor("gpu")
@pytest.mark.model("bert")
@pytest.mark.integration("smart_sifting")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smart_sifting_gpu(framework_version, ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smart_sifting(ecr_image)
    instance_type = instance_type or "ml.g4dn.12xlarge"

    source_code = SourceCode(
        source_dir=smart_sifting_path,
        entry_script="train_plt_smart_sifting.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}
    hyperparameters = {"epochs": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-smart-sifting",
        )
