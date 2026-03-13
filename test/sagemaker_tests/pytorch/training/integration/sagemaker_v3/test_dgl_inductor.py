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
from sagemaker.train.configs import SourceCode

from ...integration import resources_path, DEFAULT_TIMEOUT
from .timeout import timeout

from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer


DGL_DATA_PATH = os.path.join(resources_path, "dgl-gcn")
DGL_SCRIPT_PATH = os.path.join(DGL_DATA_PATH, "train.py")
inductor_instance_types = ["ml.g5.12xlarge", "ml.g5.12xlarge", "ml.g4dn.12xlarge"]


@pytest.mark.skip("DGL binaries are not installed in DLCs by default")
@pytest.mark.skip_gpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_inductor_test
@pytest.mark.integration("dgl")
@pytest.mark.processor("cpu")
@pytest.mark.model("gcn")
@pytest.mark.team("dgl")
def test_dgl_gcn_training_cpu(ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.c5.xlarge"

    source_code = SourceCode(
        source_dir=DGL_DATA_PATH,
        entry_script="train.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters={"inductor": 1},
            job_name="test-pt-v3-dgl-inductor",
        )


@pytest.mark.skip("DGL binaries are not installed in DLCs by default")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_inductor_test
@pytest.mark.integration("dgl")
@pytest.mark.processor("gpu")
@pytest.mark.model("gcn")
@pytest.mark.team("dgl")
@pytest.mark.parametrize("instance_type", inductor_instance_types, indirect=True)
def test_dgl_gcn_training_gpu(ecr_image, sagemaker_regions, instance_type):
    skip_if_not_v3_compatible(ecr_image)
    instance_type = instance_type or "ml.g5.8xlarge"

    source_code = SourceCode(
        source_dir=DGL_DATA_PATH,
        entry_script="train.py",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters={"inductor": 1},
            job_name="test-pt-v3-dgl-inductor",
        )
