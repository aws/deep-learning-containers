import pytest

import test.test_utils as test_utils

from test.test_utils import ec2

from test.dlc_tests.ec2.pytorch.training import common_cases
from test.dlc_tests.ec2 import smclarify_cases


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("PT22_general")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
def test_pytorch_2_2_gpu(
    pytorch_training___2__2, ec2_connection, region, gpu_only
):
    raise RuntimeError("hi")
