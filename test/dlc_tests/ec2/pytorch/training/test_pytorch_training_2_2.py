import pytest

import test.test_utils as test_utils

from test.test_utils import ec2

from test.dlc_tests.ec2.pytorch.training import common_cases
from test.dlc_tests.ec2 import smclarify_cases


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("PT22_general")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.parametrize(
    "ec2_instance_type, region", common_cases.PT_EC2_GPU_INSTANCE_TYPE_AND_REGION, indirect=True
)
def test_pytorch_2_2_gpu(
    pytorch_training___2__2
):
    raise RuntimeError("hi")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("inductor")
@pytest.mark.model("N/A")
@pytest.mark.team("training-compiler")
@pytest.mark.parametrize(
    "ec2_instance_type, region",
    common_cases.PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_AND_REGION,
    indirect=True,
)
def test_pytorch_2_2_gpu_inductor(
    pytorch_training___2__2
):
    print('hi')
