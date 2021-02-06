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
from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')

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
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if Version(image_framework_version) != Version("2.3.1") or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on TensorFlow 2.3")
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
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0",
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
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if Version(image_framework_version) != Version("2.3.1") or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on TensorFlow 2.3")
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
                                   "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0",
                                }
                           },
                           sagemaker_session=sagemaker_session,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version='py3',
                           base_job_name='smp-test2')
    estimator.fit()
