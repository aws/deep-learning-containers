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

import pytest
from sagemaker.huggingface import HuggingFace

from ...integration import ROLE, distrilbert_script


@pytest.mark.model("hf_bert")
@pytest.mark.integration("hf_local")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_distilbert_base(docker_image, processor, instance_type, sagemaker_local_session, py_version):

    # hyperparameters, which are passed into the training job
    hyperparameters = {"max_steps": 5, "train_batch_size": 4, "model_name": "distilbert-base-uncased"}

    estimator = HuggingFace(
        entry_point=distrilbert_script,
        instance_type="local_gpu",
        sagemaker_session=sagemaker_local_session,
        image_uri=docker_image,
        instance_count=1,
        role=ROLE,
        py_version=py_version,
        hyperparameters=hyperparameters,
    )

    estimator.fit()
