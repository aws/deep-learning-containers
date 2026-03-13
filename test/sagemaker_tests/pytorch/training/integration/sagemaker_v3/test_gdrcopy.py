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
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import Torchrun
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT
from .timeout import timeout
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer
from ....training import get_efa_test_instance_type

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
GDRCOPY_SANITY_TEST_CMD = os.path.join(RESOURCE_PATH, "gdrcopy", "test_gdrcopy.sh")


def validate_or_skip_gdrcopy(ecr_image):
    if not can_run_gdrcopy(ecr_image):
        pytest.skip("GDRCopy is only supported on CUDA 11.7+, and on PyTorch 1.13.1 or higher")


def can_run_gdrcopy(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.13.1") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("117")


@pytest.mark.skip(
    reason="gdrcopy sanity test in the sagemaker test job is duplicate test to the gdrcopy test in the ec2 test job"
)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_trcomp_containers
@pytest.mark.gdrcopy()
@pytest.mark.team("smdataparallel")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p4d.24xlarge"]), indirect=True
)
@pytest.mark.team("conda")
def test_sanity_gdrcopy(ecr_image, efa_instance_type, sagemaker_regions):
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_gdrcopy(ecr_image)

    source_code = SourceCode(
        source_dir=os.path.dirname(GDRCOPY_SANITY_TEST_CMD),
        entry_script=os.path.basename(GDRCOPY_SANITY_TEST_CMD),
    )
    compute_params = {"instance_type": efa_instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-gdrcopy-sanity",
        )
