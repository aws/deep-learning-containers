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

import pytest
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.serve import Mode

from ...integration import data_dir, dist_operations_path, mnist_script, ROLE
from ...utils.local_mode_utils import assert_files_exist

MODEL_SUCCESS_FILES = {
    "model": ["success"],
    "output": ["success"],
}


@pytest.fixture(scope="session", name="dist_gpu_backend", params=["gloo"])
def fixture_dist_gpu_backend(request):
    return request.param


def _create_model_trainer(
    docker_image,
    entry_point,
    sagemaker_session,
    hyperparameters,
    instance_count=1,
    instance_type="local",
    output_path=None,
):
    """Create a ModelTrainer for local mode testing."""
    source_code = SourceCode(entry_script=entry_point)

    compute = Compute(
        instance_type=instance_type,
        instance_count=instance_count,
    )

    return ModelTrainer(
        training_image=docker_image,
        source_code=source_code,
        compute=compute,
        hyperparameters=hyperparameters,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        training_mode=Mode.LOCAL_CONTAINER,
        output_path=output_path,
    )


@pytest.mark.processor("cpu")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_gpu
def test_dist_operations_path_cpu(docker_image, dist_cpu_backend, sagemaker_local_session, tmpdir):
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=dist_operations_path,
        sagemaker_session=sagemaker_local_session,
        hyperparameters={"backend": dist_cpu_backend},
        instance_count=2,
        instance_type="local",
        output_path="file://{}".format(tmpdir),
    )

    _train_and_assert_success(model_trainer, str(tmpdir))


@pytest.mark.processor("gpu")
@pytest.mark.integration("nccl")
@pytest.mark.model("unknown_model")
@pytest.mark.skip_cpu
def test_dist_operations_path_gpu_nccl(docker_image, sagemaker_local_session, tmpdir):
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=dist_operations_path,
        sagemaker_session=sagemaker_local_session,
        hyperparameters={"backend": "nccl"},
        instance_count=1,
        instance_type="local_gpu",
        output_path="file://{}".format(tmpdir),
    )

    _train_and_assert_success(model_trainer, str(tmpdir))


@pytest.mark.processor("cpu")
@pytest.mark.integration("nccl")
@pytest.mark.model("mnist")
@pytest.mark.skip_gpu
@pytest.mark.skip(
    "Skipping as NCCL is not installed on CPU image. Refer https://github.com/aws/deep-learning-containers/issues/1289"
)
def test_cpu_nccl(docker_image, sagemaker_local_session, tmpdir):
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=mnist_script,
        sagemaker_session=sagemaker_local_session,
        hyperparameters={"backend": "nccl"},
        instance_count=2,
        instance_type="local",
        output_path="file://{}".format(tmpdir),
    )

    input_data = InputData(
        channel_name="training",
        data_source="file://{}".format(os.path.join(data_dir, "training")),
    )

    with pytest.raises(RuntimeError):
        model_trainer.train(input_data_config=[input_data], wait=True)

    failure_file = {"output": ["failure"]}
    assert_files_exist(str(tmpdir), failure_file)


@pytest.mark.processor("cpu")
@pytest.mark.model("mnist")
@pytest.mark.skip_gpu
def test_mnist_cpu(docker_image, dist_cpu_backend, sagemaker_local_session, tmpdir):
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=mnist_script,
        sagemaker_session=sagemaker_local_session,
        hyperparameters={"backend": dist_cpu_backend},
        instance_count=2,
        instance_type="local",
        output_path="file://{}".format(tmpdir),
    )

    success_files = {
        "model": ["model_0.pth", "model_1.pth"],
        "output": ["success"],
    }
    _train_and_assert_success(model_trainer, str(tmpdir), success_files)


def _train_and_assert_success(model_trainer, output_path, output_files=MODEL_SUCCESS_FILES):
    input_data = InputData(
        channel_name="training",
        data_source="file://{}".format(os.path.join(data_dir, "training")),
    )
    model_trainer.train(input_data_config=[input_data], wait=True)
    assert_files_exist(output_path, output_files)
