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
"""SageMaker endpoint integration tests for Ray DLC (CPU).

Tests 5 models on ml.m5.xlarge:
  - cv-densenet, mnist-direct-app, tabular, nlp, audio-ffmpeg

All shared logic lives in common.py; this file only sets device config.
"""

import pytest

from .common import (
    make_model_endpoint_fixture,
    make_model_name_fixture,
    make_model_package_fixture,
    run_test_audio_ffmpeg,
    run_test_cv_densenet,
    run_test_mnist_direct_app,
    run_test_nlp,
    run_test_tabular,
)

DEVICE = "cpu"
INSTANCE_TYPE = "ml.m5.xlarge"

# Register fixtures for this module
model_name = make_model_name_fixture()
model_package = make_model_package_fixture(DEVICE, INSTANCE_TYPE)
model_endpoint = make_model_endpoint_fixture(DEVICE, INSTANCE_TYPE)


@pytest.mark.parametrize("model_name", ["cv-densenet"], indirect=True)
def test_cv_densenet(model_endpoint, model_name):
    run_test_cv_densenet(model_endpoint)


@pytest.mark.parametrize("model_name", ["mnist-direct-app"], indirect=True)
def test_mnist_direct_app(model_endpoint, model_name):
    run_test_mnist_direct_app(model_endpoint)


@pytest.mark.parametrize("model_name", ["tabular"], indirect=True)
def test_tabular(model_endpoint, model_name):
    run_test_tabular(model_endpoint)


@pytest.mark.parametrize("model_name", ["nlp"], indirect=True)
def test_nlp(model_endpoint, model_name):
    run_test_nlp(model_endpoint)


@pytest.mark.parametrize("model_name", ["audio-ffmpeg"], indirect=True)
def test_audio_ffmpeg(model_endpoint, model_name):
    run_test_audio_ffmpeg(model_endpoint)
