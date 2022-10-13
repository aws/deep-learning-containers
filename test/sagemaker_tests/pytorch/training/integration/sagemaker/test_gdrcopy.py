# Copyright 2018-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from sagemaker import utils
from sagemaker.instance_group import InstanceGroup

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT, throughput_path
from ...integration.sagemaker.timeout import timeout
from test.test_utils import get_framework_and_version_from_tag
from . import invoke_pytorch_estimator


def validate_or_skip_gdrcopy(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    framework_ready = Version(image_framework_version) in SpecifierSet(">=1.12.1")
    cuda_ready = Version(image_cuda_version.lstrip("cu")) >= Version("113")
    if not (framework_ready and cuda_ready):
        pytest.skip("GDRCopy is supported on CUDA 113 on PyTorch v1.12.1 and above")


@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("gdrcopy")
@pytest.mark.parametrize('instance_types', ["ml.p4d.24xlarge"])
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.efa()
def test_gdr_copy():
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_gdrcopy(ecr_image)
        hyperparameters = {
            "types": "sanity copybw"
        }
        estimator_parameter = {
            'entry_point': 'test_gdrcopy.py',
            'role': 'SageMakerRole',
            'instance_count': 1,
            'instance_type': instance_types,
            'source_dir': throughput_path,
            'framework_version': framework_version,
            'hyperparameters': hyperparameters
        }

        job_name = utils.unique_name_from_base('test-gdrcopy')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)
