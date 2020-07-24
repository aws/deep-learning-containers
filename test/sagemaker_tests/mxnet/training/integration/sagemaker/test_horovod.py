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
from sagemaker.mxnet import MXNet

from test.test_utils import MLModel
from ...integration import RESOURCE_PATH
from ...integration.utils import unique_name_from_base


@pytest.mark.multinode("multinode")
@pytest.mark.integration("horovod")
@pytest.mark.model(MLModel.MNIST.value)
def test_distributed_training_horovod(sagemaker_session,
                                      instance_type,
                                      ecr_image,
                                      tmpdir,
                                      framework_version):

    mpi_options = '-verbose -x orte_base_help_aggregate=0'
    estimator = MXNet(
        entry_point=os.path.join(RESOURCE_PATH, 'mnist', 'horovod_mnist.py'),
        role='SageMakerRole',
        train_instance_type=instance_type,
        train_instance_count=2,
        image_name=ecr_image,
        framework_version=framework_version,
        py_version='py3',
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': mpi_options,
                         'sagemaker_mpi_num_of_processes_per_host': 1},
        sagemaker_session=sagemaker_session)

    estimator.fit(job_name=unique_name_from_base('test-mx-horovod'))

    model_data_source = sagemaker.local.data.get_data_source_instance(
        estimator.model_data, sagemaker_session)

    for filename in model_data_source.get_file_list():
        assert os.path.basename(filename) == 'model.tar.gz'
