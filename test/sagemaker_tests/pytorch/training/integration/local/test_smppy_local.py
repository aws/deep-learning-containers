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

import os, sys
import subprocess

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from sagemaker.pytorch import PyTorch

from ...integration import ROLE, data_dir, smppy_mnist_script, get_framework_and_version_from_tag
from ...utils.local_mode_utils import assert_files_exist

# only the latest version of sagemaker supports profiler
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker>=2.180.0"])


def _skip_if_image_is_not_compatible_with_smppy(image_uri):
    _, framework_version = get_framework_and_version_from_tag(image_uri)
    compatible_versions = SpecifierSet(">=2.0")
    if Version(framework_version) not in compatible_versions:
        pytest.skip(f"This test only works for PT versions in {compatible_versions}")


@pytest.mark.usefixtures("feature_smppy_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_cpu
def test_smppy_mnist_local(docker_image, sagemaker_local_session, tmpdir):
    _skip_if_image_is_not_compatible_with_smppy(docker_image)
    estimator = PyTorch(
        entry_point=smppy_mnist_script,
        role=ROLE,
        image_uri=docker_image,
        instance_count=1,
        instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        output_path="file://{}".format(tmpdir),
        hyperparameters={"epochs": 1},
    )

    _train_and_assert_success(
        estimator, str(tmpdir), {"training": "file://{}".format(os.path.join(data_dir, "training"))}
    )


def _train_and_assert_success(estimator, output_path, fit_params={}):
    estimator.fit(fit_params)
    success_files = {"output": ["success"]}
    assert_files_exist(output_path, success_files)
