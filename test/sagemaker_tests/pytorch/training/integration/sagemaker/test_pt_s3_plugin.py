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

import boto3
import pytest
from sagemaker.pytorch import PyTorch
from sagemaker import utils
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import (DEFAULT_TIMEOUT, resnet18_path)
from ...integration.sagemaker.timeout import timeout
from test.test_utils import get_framework_and_version_from_tag

MULTI_GPU_INSTANCE = 'ml.p3.8xlarge'
CPU_INSTANCE = 'ml.c5.4xlarge'

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


def can_run_s3_plugin(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.7")


def validate_or_skip_s3_plugin(ecr_image):
    if not can_run_s3_plugin(ecr_image):
        pytest.skip("S3 plugin is added only on PyTorch 1.7 or higher")


@pytest.mark.processor("gpu")
@pytest.mark.integration("pt_s3_plugin_gpu")
@pytest.mark.model("resnet18")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_pt_s3_plugin_sm_gpu(sagemaker_session, framework_version, ecr_image):
    validate_or_skip_s3_plugin(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="main.py",
            source_dir=resnet18_path,
            image_uri=ecr_image,
            role='SageMakerRole',
            instance_count=1,
            instance_type=MULTI_GPU_INSTANCE,
            sagemaker_session=sagemaker_session,
            framework_version=framework_version
        )
        job_name = utils.unique_name_from_base('test-pytorch-s3-plugin-gpu')
        pytorch.fit(job_name=job_name)


@pytest.mark.processor("cpu")
@pytest.mark.integration("pt_s3_plugin_cpu")
@pytest.mark.model("resnet18")
@pytest.mark.skip_gpu
@pytest.mark.skip_py2_containers
def test_pt_s3_plugin_sm_cpu(sagemaker_session, framework_version, ecr_image):
    validate_or_skip_s3_plugin(ecr_image)
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="main.py",
            source_dir=resnet18_path,
            image_uri=ecr_image,
            role='SageMakerRole',
            instance_count=1,
            instance_type=CPU_INSTANCE,
            sagemaker_session=sagemaker_session,
            framework_version=framework_version
        )
        job_name = utils.unique_name_from_base('test-pytorch-s3-plugin-cpu')
        pytorch.fit(job_name=job_name)

