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
"""EC2 container integration tests for Ray DLC (CPU).

Tests 5 models via local docker run (no GPU flags):
  - cv-densenet, mnist-direct-app, tabular, nlp, audio-ffmpeg

All shared logic lives in common.py; this file only sets device config.
"""

import pytest
from ray.ec2.common import (
    make_container_fixture,
    make_model_name_fixture,
    run_test_audio_ffmpeg,
    run_test_cv_densenet,
    run_test_mnist_direct_app,
    run_test_nlp,
    run_test_tabular,
)

DEVICE = "cpu"

# Register fixtures for this module
model_name = make_model_name_fixture()
container = make_container_fixture(DEVICE)


@pytest.mark.parametrize("model_name", ["cv-densenet"], indirect=True)
def test_cv_densenet(container, model_name):
    run_test_cv_densenet(container)


@pytest.mark.parametrize("model_name", ["mnist-direct-app"], indirect=True)
def test_mnist_direct_app(container, model_name):
    run_test_mnist_direct_app(container)


@pytest.mark.parametrize("model_name", ["tabular"], indirect=True)
def test_tabular(container, model_name):
    run_test_tabular(container, check_packages=True)


@pytest.mark.parametrize("model_name", ["nlp"], indirect=True)
def test_nlp(container, model_name):
    run_test_nlp(container)


@pytest.mark.parametrize("model_name", ["audio-ffmpeg"], indirect=True)
def test_audio_ffmpeg(container, model_name):
    run_test_audio_ffmpeg(container)
