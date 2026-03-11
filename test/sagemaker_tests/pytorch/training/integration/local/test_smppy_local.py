# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, InputData, Compute
from sagemaker.serve import Mode

from ...integration import ROLE, data_dir, smppy_mnist_script, get_framework_and_version_from_tag
from ...utils.local_mode_utils import assert_files_exist


def _skip_if_image_is_not_compatible_with_smppy(image_uri):
    _, framework_version = get_framework_and_version_from_tag(image_uri)
    compatible_versions = SpecifierSet(">=2.0,<2.7.1")
    if Version(framework_version) not in compatible_versions:
        pytest.skip(f"This test only works for PT versions in {compatible_versions}")


def _create_model_trainer(
    docker_image,
    entry_point,
    sagemaker_session,
    instance_type="local_gpu",
    hyperparameters=None,
    output_path=None,
):
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


@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_cpu
def test_smppy_mnist_local(docker_image, sagemaker_local_session, tmpdir):
    _skip_if_image_is_not_compatible_with_smppy(docker_image)

    model_trainer = _create_model_trainer(
        docker_image=docker_image,
        entry_point=smppy_mnist_script,
        sagemaker_session=sagemaker_local_session,
        instance_type="local_gpu",
        hyperparameters={"epochs": 1},
        output_path="file://{}".format(tmpdir),
    )

    input_data = InputData(
        channel_name="training",
        data_source="file://{}".format(os.path.join(data_dir, "training")),
    )

    _train_and_assert_success(model_trainer, str(tmpdir), input_data_config=[input_data])


def _train_and_assert_success(model_trainer, output_path, input_data_config=None):
    model_trainer.train(input_data_config=input_data_config, wait=True)
    success_files = {"output": ["success"]}
    assert_files_exist(output_path, success_files)
