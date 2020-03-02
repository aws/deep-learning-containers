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

import json
import os
import tarfile

import pytest
from sagemaker.tensorflow import TensorFlow

from test.integration.utils import processor, py_version  # noqa: F401

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


@pytest.mark.skip_gpu
@pytest.mark.parametrize('instances, processes', [
    [1, 2],
    (2, 1),
    (2, 2),
    (5, 2)])
def test_distributed_training_horovod_basic(instances,
                                            processes,
                                            sagemaker_local_session,
                                            docker_image,
                                            tmpdir,
                                            framework_version):
    output_path = 'file://%s' % tmpdir
    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, 'hvdbasic', 'train_hvd_basic.py'),
        role='SageMakerRole',
        train_instance_type='local',
        sagemaker_session=sagemaker_local_session,
        train_instance_count=instances,
        image_name=docker_image,
        output_path=output_path,
        framework_version=framework_version,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_network_interface_name': 'eth0',
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    estimator.fit('file://{}'.format(os.path.join(RESOURCE_PATH, 'mnist', 'data-distributed')))

    tmp = str(tmpdir)
    extract_files(output_path.replace('file://', ''), tmp)

    size = instances * processes

    for rank in range(size):
        local_rank = rank % processes
        assert read_json('local-rank-%s-rank-%s' % (local_rank, rank), tmp) == {
            'local-rank': local_rank, 'rank': rank, 'size': size}


def read_json(file, tmp):
    with open(os.path.join(tmp, file)) as f:
        return json.load(f)


def assert_files_exist_in_tar(output_path, files):
    if output_path.startswith('file://'):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, 'model.tar.gz')
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)


def extract_files(output_path, tmpdir):
    with tarfile.open(os.path.join(output_path, 'model.tar.gz')) as tar:
        tar.extractall(tmpdir)
