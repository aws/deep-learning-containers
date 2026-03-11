# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.train.configs import SourceCode, Compute
from sagemaker.train.distributed import Torchrun

from ...integration import DEFAULT_TIMEOUT, mnist_path
from ...integration.sagemaker.timeout import timeout
from ....training import get_efa_test_instance_type
from . import invoke_pytorch_training
from .test_torch_distributed import validate_or_skip_distributed_training


@pytest.mark.skipif(
    os.getenv("SM_EFA_TEST_INSTANCE_TYPE") == "ml.p5.48xlarge",
    reason="Low availability of instance type; Must ensure test works on new instances.",
)
@pytest.mark.skip_pytorchddp_test
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_inductor_test
@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("pytorchddp")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p4d.24xlarge"]), indirect=True
)
@pytest.mark.efa()
@pytest.mark.team("training-compiler")
def test_pytorchddp_throughput_gpu(
    framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_distributed_training(ecr_image)

        source_code = SourceCode(
            entry_script="pytorchddp_throughput_mnist.py",
            source_dir=mnist_path,
        )

        compute = Compute(
            instance_type=efa_instance_type,
            instance_count=2,
        )

        hyperparameters = {"inductor": 1}

        invoke_pytorch_training(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute=compute,
            hyperparameters=hyperparameters,
            distributed_runner=Torchrun(),
            job_name="test-pytorchddp-throughput-gpu",
        )
