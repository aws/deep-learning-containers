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

from sagemaker.pytorch import PyTorch
from sagemaker import utils

from .timeout import timeout
from ...integration import smart_sifting_path, DEFAULT_TIMEOUT
from .... import invoke_pytorch_helper_function
from test.test_utils import get_framework_and_version_from_tag


def validate_or_skip_smart_sifting(ecr_image):
    if not can_run_smart_sifting(ecr_image):
        pytest.skip("Smart sifting is only available for use with PT 2.0.1")


def can_run_smart_sifting(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet("==2.0.1")


@pytest.mark.usefixtures("feature_smart_sifting_present")
@pytest.mark.processor("cpu")
@pytest.mark.model("bert")
@pytest.mark.integration("smart_sifting")
@pytest.mark.skip_gpu
@pytest.mark.skip_py2_containers
def test_smart_sifting_cpu(framework_version, ecr_image, sagemaker_regions, instance_type):
    validate_or_skip_smart_sifting(ecr_image)
    instance_type = instance_type or "ml.c4.xlarge"
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
    }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_smart_sifting, function_args)


@pytest.mark.usefixtures("feature_smart_sifting_present")
@pytest.mark.processor("gpu")
@pytest.mark.model("bert")
@pytest.mark.integration("smart_sifting")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smart_sifting_gpu(framework_version, ecr_image, sagemaker_regions, instance_type):
    validate_or_skip_smart_sifting(ecr_image)
    instance_type = instance_type or "ml.g4dn.12xlarge"
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
    }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_smart_sifting, function_args)


def _test_smart_sifting(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type=None,
    instance_count=1,
):
    est_params = {
        "entry_point": "train_plt_smart_sifting.py",
        "source_dir": smart_sifting_path,
        "role": "SageMakerRole",
        "sagemaker_session": sagemaker_session,
        "image_uri": ecr_image,
        "framework_version": framework_version,
        "hyperparameters": {"epochs": 1},
    }
    est_params["instance_type"] = instance_type
    est_params["instance_count"] = instance_count
    job_name = "test-smart-sifting-plt"
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(**est_params)
        pytorch.fit(job_name=utils.unique_name_from_base(job_name))
