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
import boto3
from datetime import datetime, timedelta

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from sagemaker import utils

from ...integration import DEFAULT_TIMEOUT, mnist_path
from ...integration.sagemaker.timeout import timeout
from ....training import get_efa_test_instance_type
from test.test_utils import get_framework_and_version_from_tag
from . import invoke_pytorch_estimator

METRIC_DEFINITIONS_SMDDP = [
    {'Name': 'SMDDP-COLLECTIVES-VERSION', 'Regex': 'SMDDP: Running SMDDPCollectives v1.0.(.*?)'},
    # {'Name': 'SMDDP-COLLECTIVES-ENV-CHECK', 'Regex': 'SMDDP: Environment checks succeeded.(.*?)\n'}
]
METRIC_DEFINITIONS_NCCL = [
    {'Name': 'SMDDP-COLLECTIVES-VERSION', 'Regex': 'SMDDP: Running SMDDPCollectives v(.*?)\n'}
]


def _validate_or_skip_pytorchddp_backend(ecr_image: str):
    if not _is_smddpcoll_supported_image(ecr_image):
        pytest.skip("SMDDP-Collectives is supported on PyTorch v1.12.1 and above")


def _is_smddpcoll_supported_image(ecr_image: str) -> bool:
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.12.1")


def _is_smddpcoll_supported_instance(instance_type: str) -> bool:
    supported_instances = ['ml.p4d.24xlarge']
    return instance_type in supported_instances


def _fetch_metrics(cloudwatch_client, job_name: str, use_smddp_metrics: bool) -> dict:
    metric_values = {}
    metric_definitions = METRIC_DEFINITIONS_SMDDP if use_smddp_metrics else METRIC_DEFINITIONS_NCCL
    for definition in metric_definitions:
        datapoints = cloudwatch_client.get_metric_statistics(
            Namespace='/aws/sagemaker/TrainingJobs',
            MetricName=definition['Name'],
            Dimensions=[{'Name': 'Host', 'Value': job_name}],
            StartTime=datetime.now() - timedelta(days=1), EndTime=datetime.now(),
            Period=3600, Statistics=['SampleCount']
        )['Datapoints']
        if datapoints:
            metric_values[definition['Name']] = round(datapoints[0]['SampleCount'])
    return metric_values


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("pytorchddp_backend")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p3.16xlarge", "ml.p4d.24xlarge"]), indirect=True)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.efa()
def test_pytorchddp_backend_default_gpu(framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    with timeout(minutes=DEFAULT_TIMEOUT):
        _validate_or_skip_pytorchddp_backend(ecr_image)
        distribution = {'pytorchddp': {'enabled': True}}
        estimator_parameter = {
            'entry_point': 'pytorchddp_throughput_mnist.py',
            'role': 'SageMakerRole',
            'instance_count': 2,
            'instance_type': efa_instance_type,
            'source_dir': mnist_path,
            'framework_version': framework_version,
            'distribution': distribution,
            'metric_definitions': METRIC_DEFINITIONS_SMDDP
        }

        job_name = utils.unique_name_from_base('test-pytorchddp-backend-default-gpu')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)

        cloudwatch_client = boto3.client('cloudwatch')
        for _ in sagemaker_regions:
            metric_values = _fetch_metrics(cloudwatch_client, job_name, True)
            criterion = int(_is_smddpcoll_supported_instance(efa_instance_type))
            assert metric_values['SMDDP-COLLECTIVES-VERSION'] == criterion
            # assert metric_values['SMDDP-COLLECTIVES-ENV-CHECK'] == criterion


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("pytorchddp_backend")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p3.16xlarge", "ml.p4d.24xlarge"]), indirect=True)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.efa()
def test_pytorchddp_backend_auto_gpu(framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    with timeout(minutes=DEFAULT_TIMEOUT):
        _validate_or_skip_pytorchddp_backend(ecr_image)
        distribution = {'pytorchddp': {'enabled': True, 'communication_options': {'backend': 'auto'}}}
        estimator_parameter = {
            'entry_point': 'pytorchddp_throughput_mnist.py',
            'role': 'SageMakerRole',
            'instance_count': 2,
            'instance_type': efa_instance_type,
            'source_dir': mnist_path,
            'framework_version': framework_version,
            'distribution': distribution,
            'metric_definitions': METRIC_DEFINITIONS_SMDDP
        }

        job_name = utils.unique_name_from_base('test-pytorchddp-backend-auto-gpu')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)

        cloudwatch_client = boto3.client('cloudwatch')
        for _ in sagemaker_regions:
            metric_values = _fetch_metrics(cloudwatch_client, job_name, True)
            criterion = int(_is_smddpcoll_supported_instance(efa_instance_type))
            assert metric_values['SMDDP-COLLECTIVES-VERSION'] == criterion
            # assert metric_values['SMDDP-COLLECTIVES-ENV-CHECK'] == criterion


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("pytorchddp_backend")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p3.16xlarge", "ml.p4d.24xlarge"]), indirect=True)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.efa()
def test_pytorchddp_backend_nccl_gpu(framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    with timeout(minutes=DEFAULT_TIMEOUT):
        _validate_or_skip_pytorchddp_backend(ecr_image)
        distribution = {'pytorchddp': {'enabled': True, 'communication_options': {'backend': 'nccl'}}}
        
        estimator_parameter = {
            'entry_point': 'pytorchddp_throughput_mnist.py',
            'role': 'SageMakerRole',
            'instance_count': 2,
            'instance_type': efa_instance_type,
            'source_dir': mnist_path,
            'framework_version': framework_version,
            'distribution': distribution,
            'metric_definitions': METRIC_DEFINITIONS_NCCL
        }

        job_name = utils.unique_name_from_base('test-pytorchddp-backend-nccl-gpu')
        invoke_pytorch_estimator(ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name)

        cloudwatch_client = boto3.client('cloudwatch')
        for _ in sagemaker_regions:
            metric_values = _fetch_metrics(cloudwatch_client, job_name, False)
            assert metric_values['SMDDP-COLLECTIVES-VERSION'] == 0
