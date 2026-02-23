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

from ...utils.local_mode_utils import assert_files_exist
from ...integration import (
    data_dir,
    fastai_path,
    fastai_mnist_script,
    mnist_path,
    mnist_script,
    ROLE,
    get_framework_and_version_from_tag,
)
from packaging.version import Version
from packaging.specifiers import SpecifierSet


def _create_model_trainer(docker_image, entry_point, sagemaker_session, 
                          instance_type="local", hyperparameters=None, output_path=None):
    """Create a ModelTrainer for local mode testing."""
    source_code = SourceCode(entry_script=entry_point)
    
    compute = Compute(
        instance_type=instance_type,
        instance_count=1,
    )
    
    return ModelTrainer(
        training_image=docker_image,
        source_code=source_code,
        compute=compute,
        hyperparameters=hyperparameters or {},
        role=ROLE,
        sagemaker_session=sagemaker_session,
        training_mode=Mode.LOCAL_CONTAINER,
        output_path=output_path,
    )


@pytest.mark.model("mnist")
def test_mnist(docker_image, processor, instance_type, sagemaker_local_session, tmpdir):
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=mnist_script,
        sagemaker_session=sagemaker_local_session,
        instance_type=instance_type,
        hyperparameters={"processor": processor},
        output_path="file://{}".format(tmpdir),
    )

    input_data = InputData(
        channel_name="training",
        data_source="file://{}".format(os.path.join(data_dir, "training")),
    )

    _train_and_assert_success(
        model_trainer,
        str(tmpdir),
        input_data_config=[input_data],
        model_pth="model_0.pth",
    )


@pytest.mark.integration("fastai")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
def test_fastai_mnist(docker_image, instance_type, py_version, sagemaker_local_session, tmpdir):
    _, image_framework_version = get_framework_and_version_from_tag(docker_image)
    if Version(image_framework_version) in SpecifierSet(">=1.9,<1.13"):
        pytest.skip("Fast ai is not supported on PyTorch v1.9.x, v1.10.x, v1.11.x, v1.12.x")
    if Version(image_framework_version) in SpecifierSet("~=2.6.0"):
        pytest.skip("Fast ai doesn't release for PyTorch v2.6.x")
    
    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=fastai_mnist_script,
        sagemaker_session=sagemaker_local_session,
        instance_type=instance_type,
        output_path="file://{}".format(tmpdir),
    )

    _train_and_assert_success(model_trainer, str(tmpdir))


def _train_and_assert_success(model_trainer, output_path, input_data_config=None, model_pth="model.pth"):
    model_trainer.train(input_data_config=input_data_config, wait=True)

    success_files = {"model": [model_pth], "output": ["success"]}
    assert_files_exist(output_path, success_files)
