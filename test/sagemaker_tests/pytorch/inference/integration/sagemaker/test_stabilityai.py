from __future__ import absolute_import

import sys

import pytest
import logging


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.stabilityai_only
def test_mnist_distributed_gpu_stabilityai_one(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    raise Exception("Deliberate error 1")
