# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sagemaker
from sagemaker.tensorflow import TensorFlow
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


def validate_or_skip_smmodelparallel(ecr_image):
    if not can_run_smmodelparallel(ecr_image):
        pytest.skip("Model Parallelism is supported on CUDA 11 on TensorFlow v2.3.1 or higher")


def can_run_smmodelparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=2.3.1") and Version(
        image_cuda_version.strip("cu")) >= Version("110")


def validate_or_skip_smmodelparallel_efa(ecr_image):
    if not can_run_smmodelparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on TensorFlow v2.4.1 or higher")


def can_run_smmodelparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=2.4.1") and Version(image_cuda_version.strip("cu")) >= Version("110")


@pytest.mark.integration("smmodelparallel")
@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("tf2_conv.py", 2), ("tf2_conv_xla.py", 2), ("smmodelparallel_hvd2_conv.py", 4), ("send_receive_checkpoint.py", 2), ("tf2_checkpoint_test.py", 2)])
@pytest.mark.efa()
def test_smmodelparallel_efa(sagemaker_session, efa_instance_type, ecr_image, tmpdir, framework_version, test_script, num_processes):
    """
    Tests SM Modelparallel in sagemaker
    """
    validate_or_skip_smmodelparallel_efa(ecr_image)
    smmodelparallel_path = os.path.join(RESOURCE_PATH, 'smmodelparallel')
    estimator = TensorFlow(entry_point=test_script,
                           role='SageMakerRole',
                           instance_count=1,
                           instance_type=efa_instance_type,
                           source_dir=smmodelparallel_path,
                           distributions={
                               "mpi": {
                                   "enabled": True,
                                   "processes_per_host": num_processes,
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa ",
                                }
                           },
                           sagemaker_session=sagemaker_session,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version='py3',
                           base_job_name='smp-test1')
    estimator.fit()


@pytest.mark.integration("smmodelparallel")
@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_hvd2_conv_multinode.py", 2)])
@pytest.mark.efa()
def test_smmodelparallel_multinode_efa(sagemaker_session, efa_instance_type, ecr_image, tmpdir, framework_version, test_script, num_processes):
    """
    Tests SM Modelparallel in sagemaker
    """
    validate_or_skip_smmodelparallel_efa(ecr_image)
    smmodelparallel_path = os.path.join(RESOURCE_PATH, 'smmodelparallel')
    estimator = TensorFlow(entry_point=test_script,
                           role='SageMakerRole',
                           instance_count=2,
                           instance_type=efa_instance_type,
                           source_dir=smmodelparallel_path,
                           distributions={
                               "mpi": {
                                   "enabled": True,
                                   "processes_per_host": num_processes,
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa ",
                                }
                           },
                           sagemaker_session=sagemaker_session,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version='py3',
                           base_job_name='smp-test2')
    estimator.fit()


@pytest.mark.integration("smmodelparallel")
@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("tf2_conv.py", 2), ("tf2_conv_xla.py", 2), ("smmodelparallel_hvd2_conv.py", 4), ("send_receive_checkpoint.py", 2), ("tf2_checkpoint_test.py", 2)])
def test_smmodelparallel(sagemaker_session, instance_type, ecr_image, tmpdir, framework_version, test_script, num_processes):
    """
    Tests SM Modelparallel in sagemaker
    """
    instance_type = "ml.p3.16xlarge"
    validate_or_skip_smmodelparallel(ecr_image)
    smmodelparallel_path = os.path.join(RESOURCE_PATH, 'smmodelparallel')
    estimator = TensorFlow(entry_point=test_script,
                           role='SageMakerRole',
                           instance_count=1,
                           instance_type=instance_type,
                           source_dir=smmodelparallel_path,
                           distributions={
                               "mpi": {
                                   "enabled": True,
                                   "processes_per_host": num_processes,
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 ",
                                }
                           },
                           sagemaker_session=sagemaker_session,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version='py3',
                           base_job_name='smp-test1')
    estimator.fit()


@pytest.mark.integration("smmodelparallel")
@pytest.mark.processor("gpu")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize("test_script, num_processes", [("smmodelparallel_hvd2_conv_multinode.py", 2)])
def test_smmodelparallel_multinode(sagemaker_session, instance_type, ecr_image, tmpdir, framework_version, test_script, num_processes):
    """
    Tests SM Modelparallel in sagemaker
    """
    instance_type = "ml.p3.16xlarge"
    validate_or_skip_smmodelparallel(ecr_image)
    smmodelparallel_path = os.path.join(RESOURCE_PATH, 'smmodelparallel')
    estimator = TensorFlow(entry_point=test_script,
                           role='SageMakerRole',
                           instance_count=2,
                           instance_type=instance_type,
                           source_dir=smmodelparallel_path,
                           distributions={
                               "mpi": {
                                   "enabled": True,
                                   "processes_per_host": num_processes,
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 ",
                                }
                           },
                           sagemaker_session=sagemaker_session,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version='py3',
                           base_job_name='smp-test2')
    estimator.fit()
