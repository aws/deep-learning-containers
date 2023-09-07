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
from sagemaker.tensorflow import TensorFlow

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
MNIST_PATH = os.path.join(RESOURCE_PATH, "mnist")


@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smppy_mnist_local(docker_image, sagemaker_local_session, tmpdir):
    script = os.path.join(MNIST_PATH, "mnist_smppy.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        image_uri=docker_image,
        output_path="file://{}".format(tmpdir),
    )

    estimator.fit("file://{}".format(os.path.join(MNIST_PATH, "data")))
