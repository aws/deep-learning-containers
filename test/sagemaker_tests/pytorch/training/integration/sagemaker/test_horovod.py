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
from sagemaker.pytorch import PyTorch

from ...integration import resources_path, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout


@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.integration("horovod")
@pytest.mark.skip_cpu
def test_horovod_simple(
    instance_type,
    sagemaker_session,
    ecr_image,
    tmpdir,
):
    instances, processes = 2, 1

    default_bucket = sagemaker_session.default_bucket()
    output_path = "s3://" + os.path.join(default_bucket, "pytorch/horovod")

    estimator = PyTorch(
        entry_point=os.path.join(resources_path, "horovod", "simple.py"),
        role="SageMakerRole",
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        train_instance_count=instances,
        image_name=ecr_image,
        output_path=output_path,
        hyperparameters={
            "sagemaker_mpi_enabled": True,
            "sagemaker_mpi_num_of_processes_per_host": processes,
        },
    )

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator.fit()

    bucket, key_prefix = estimator.model_data.replace("s3://", "").split("/", 1)
    sagemaker_session.download_data(
        path=str(tmpdir), bucket=bucket, key_prefix=key_prefix
    )

    with tarfile.open(os.path.join(str(tmpdir), "model.tar.gz")) as tar:
        tar.extractall(tmpdir)

    size = instances * processes

    for rank in range(size):
        local_rank = rank % processes
        # The simple.py script should create a JSON file with this name
        filename = "local-rank-%s-rank-%s.json" % (local_rank, rank)

        with open(os.path.join(str(tmpdir), filename)) as file:
            actual = json.load(file)
        expected = {"local-rank": local_rank, "rank": rank, "size": size}

        assert actual == expected


@pytest.mark.processor("gpu")
@pytest.mark.multinode(2)
@pytest.mark.integration("horovod")
@pytest.mark.skip_cpu
def test_horovod_training(
    instance_type,
    sagemaker_session,
    ecr_image,
    tmpdir,
):
    estimator = PyTorch(
        entry_point=os.path.join(resources_path, "horovod", "train.py"),
        role="SageMakerRole",
        train_instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        train_instance_count=2,
        image_name=ecr_image,
        hyperparameters={
            "sagemaker_mpi_enabled": True,
            "sagemaker_mpi_num_of_processes_per_host": 1,
            "epochs": 1,
        },
    )

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator.fit()
