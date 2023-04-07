"""
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import os
import boto3
import pytest
import tarfile, subprocess

from sagemaker.instance_group import InstanceGroup
from sagemaker.pytorch import PyTorch
from sagemaker import utils
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from ...integration import training_dir, DEFAULT_TIMEOUT, inductor_path
from ...integration.sagemaker.timeout import timeout
from .... import invoke_pytorch_helper_function
# from .test_inductor_helper import put_metric_data, read_metric

instance_types=("ml.p3.2xlarge", "ml.g5.4xlarge", "ml.g4dn.4xlarge")

@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("instance_type", [instance_types], indirect=True)
def test_inductor_huggingface(framework_version, ecr_image, sagemaker_regions, instance_type, tmpdir):
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'suites': "huggingface",
            'tmpdir': tmpdir,
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_inductor_performance, function_args)

@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
@pytest.mark.model("timm")
@pytest.mark.parametrize("instance_type", [instance_types], indirect=True)
def test_inductor_timm(framework_version, ecr_image, sagemaker_regions, instance_type, tmpdir):
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'suites': "timm",
            'tmpdir': tmpdir,
        }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_inductor_performance, function_args)

@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_inductor_test
@pytest.mark.model("torchbench")
@pytest.mark.parametrize("instance_type", [instance_types], indirect=True)
def test_inductor_torchbench(framework_version, ecr_image, sagemaker_regions, instance_type, tmpdir):
    function_args = {
            'framework_version': framework_version,
            'instance_type': instance_type,
            'suites': "torchbench",
            'tmpdir': tmpdir,
        }

    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_inductor_performance, function_args)

def _test_inductor_performance(ecr_image, sagemaker_session, framework_version, instance_type, suites, tmpdir):
    # with timeout(minutes=DEFAULT_TIMEOUT):   
    output_path = f"s3://sagemaker-inductor-test/{suites}"
    # job_name = "test-pt-performance-inductor-huggingface-1680825282-71e4"
    # pytorch = PyTorch(
    #     entry_point="test_inductor_helper.py",
    #     source_dir=inductor_path,
    #     role='SageMakerRole',
    #     instance_count=1,
    #     instance_type=instance_type,
    #     sagemaker_session=sagemaker_session,
    #     image_uri=ecr_image,
    #     framework_version=framework_version,
    #     hyperparameters = {'output_path':output_path,
    #                         'job_name':job_name,
    #                         'instance_type':instance_type,
    #                         # 'tmpdir': tmpdir,
    #                         'suites':suites
    #                     },
    # )
    # pytorch.fit(job_name=utils.unique_name_from_base(f'upload-pt-performance-inductor-{suites}'))
    pytorch = PyTorch(
        entry_point=f"run_inductor_{suites}.sh",
        source_dir=inductor_path,
        role='SageMakerRole',
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
        framework_version=framework_version,
        output_path=output_path,
    )
    pytorch.fit(job_name=utils.unique_name_from_base(f'test-pt-performance-inductor-{suites}'))
    job_name = pytorch.latest_training_job.name
    pytorch = PyTorch(
        entry_point="test_inductor_helper.py",
        source_dir=inductor_path,
        role='SageMakerRole',
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
        framework_version=framework_version,
        hyperparameters = {'output_path':output_path,
                            'job_name':job_name,
                            'instance_type':instance_type,
                            'suites':suites,
                        },
    )
    pytorch.fit(job_name=utils.unique_name_from_base(f'upload-pt-performance-inductor-{suites}'))

