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

from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401
from ... import invoke_tensorflow_estimator

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


@pytest.mark.integration("horovod")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
def test_distributed_training_horovod(sagemaker_session,
                                      n_virginia_sagemaker_session,
                                      instance_type,
                                      ecr_image,
                                      n_virginia_ecr_image,
                                      tmpdir,
                                      framework_version,
                                      multi_region_support):

    mpi_options = '-verbose -x orte_base_help_aggregate=0'
    estimator_parameter = {
            'entry_point': os.path.join(RESOURCE_PATH, 'mnist', 'horovod_mnist.py'),
            'role': 'SageMakerRole',
            'instance_type': instance_type,
            'instance_count': 2,
            'framework_version': framework_version,
            'py_version': 'py3',
            'script_mode': True,
            'hyperparameters': {'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': mpi_options,
                         'sagemaker_mpi_num_of_processes_per_host': 1},
        }
    estimator = invoke_tensorflow_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support)

    estimator.fit(job_name=unique_name_from_base('test-tf-horovod'))

    model_data_source = sagemaker.local.data.get_data_source_instance(
        estimator.model_data, sagemaker_session)

    for filename in model_data_source.get_file_list():
        assert os.path.basename(filename) == 'model.tar.gz'


@pytest.mark.integration("horovod")
@pytest.mark.multinode(2)
@pytest.mark.model("unknown_model")
def test_distributed_training_horovod_with_env_vars(
        sagemaker_session, n_virginia_sagemaker_session, instance_type, ecr_image, n_virginia_ecr_image, tmpdir, framework_version, multi_region_support
):

    mpi_options = "-verbose -x orte_base_help_aggregate=0"
    estimator_parameter = {
            'entry_point': os.path.join(RESOURCE_PATH, "hvdbasic", "train_hvd_env_vars.py"),
            'role': 'SageMakerRole',
            'instance_type': instance_type,
            'instance_count': 2,
            'framework_version': framework_version,
            'py_version': 'py3',
            'script_mode': True,
            'hyperparameters': {'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': mpi_options,
                         'sagemaker_mpi_num_of_processes_per_host': 2},
        }
    estimator = invoke_tensorflow_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support)

    estimator.fit(job_name=unique_name_from_base("test-tf-horovod-env-vars"))
