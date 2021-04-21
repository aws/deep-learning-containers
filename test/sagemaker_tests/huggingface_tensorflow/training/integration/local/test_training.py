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
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace
from ...integration.utils import processor, py_version  # noqa: F401

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
BERT_PATH = os.path.join(RESOURCE_PATH, 'local_scripts')


@pytest.mark.processor("gpu")
@pytest.mark.model("hf_distilbert")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_hf_tf_dummy(sagemaker_local_session, docker_image, framework_version):
    # hyperparameters, which are passed into the training job
    pass