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
import boto3, sagemaker
import pytest
import tarfile, subprocess

from sagemaker.pytorch import PyTorch
from sagemaker import utils
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from test.dlc_tests.benchmark.sagemaker import inductor_path
from test.sagemaker_tests.pytorch import invoke_pytorch_helper_function
from test.test_utils import get_framework_and_version_from_tag

instance_types = ["ml.p3.2xlarge", "ml.g5.4xlarge", "ml.g4dn.4xlarge"]


@pytest.fixture
def framework_version(pytorch_training):
    _, version = get_framework_and_version_from_tag(pytorch_training)
    return version


@pytest.fixture(autouse=True)
def inductor_support_gpu_only(framework_version, pytorch_training):
    if Version(framework_version) in SpecifierSet("<2.0.0"):
        pytest.skip("Inductor support PyTorch version >= 2.0.0")
    if "gpu" not in pytorch_training:
        pytest.skip("Inductor benchmark is only available for GPUs")


@pytest.fixture
def sagemaker_session(region):
    return sagemaker.Session(boto_session=boto3.Session(region_name=region))


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("instance_type", instance_types, indirect=True)
def test_inductor_huggingface(
    framework_version, pytorch_training, region, sagemaker_session, instance_type
):
    instance_type = instance_type
    suites = "huggingface"
    _test_inductor_performance(
        pytorch_training, sagemaker_session, framework_version, instance_type, suites
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("timm")
@pytest.mark.parametrize("instance_type", instance_types, indirect=True)
def test_inductor_timm(
    framework_version, pytorch_training, region, sagemaker_session, instance_type
):
    instance_type = instance_type
    suites = "timm"
    _test_inductor_performance(
        pytorch_training, sagemaker_session, framework_version, instance_type, suites
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("torchbench")
@pytest.mark.parametrize("instance_type", instance_types, indirect=True)
def test_inductor_torchbench(
    framework_version, pytorch_training, region, sagemaker_session, instance_type
):
    instance_type = instance_type
    suites = "torchbench"
    _test_inductor_performance(
        pytorch_training, sagemaker_session, framework_version, instance_type, suites
    )


def _test_inductor_performance(
    image_uri, sagemaker_session, framework_version, instance_type, suites
):
    output_path = f"s3://sm-inductor-test/{suites}"
    pytorch = PyTorch(
        entry_point=f"run_inductor_{suites}.sh",
        source_dir=inductor_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=framework_version,
        output_path=output_path,
        image_uri=image_uri,
        debugger_hook_config=None,
        disable_profiler=True,
        environment=dict(framework_version=framework_version),
        max_retry_attempts=5,
    )
    pytorch.fit(job_name=utils.unique_name_from_base(f"test-pt-performance-inductor-{suites}"))
    job_name = pytorch.latest_training_job.name
    pytorch = PyTorch(
        entry_point="test_inductor_helper.py",
        source_dir=inductor_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        framework_version=framework_version,
        image_uri=image_uri,
        hyperparameters={
            "output_path": output_path,
            "job_name": job_name,
            "instance_type": instance_type,
            "suites": suites,
        },
        debugger_hook_config=None,
        disable_profiler=True,
        max_retry_attempts=5,
    )
    pytorch.fit(job_name=utils.unique_name_from_base(f"upload-pt-performance-inductor-{suites}"))
