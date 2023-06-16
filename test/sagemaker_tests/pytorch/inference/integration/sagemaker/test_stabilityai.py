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
def test_mnist_distributed_gpu_stabilityai(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    raise Exception("Deliberate error 1")


@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.skip_stabilityai
def test_mnist_distributed_gpu_stabilityai(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    raise Exception("Deliberate error 2")


@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_mnist_distributed_gpu_stabilityai(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    raise Exception("Deliberate error 3")
