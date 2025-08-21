import os, sys
import logging
from typing import List

from test.test_utils import get_dlc_images, is_pr_context
from test.vllm.eks.eks_test import test_vllm_on_eks
from test.vllm.ec2.infra.setup_ec2 import setup
from test.vllm.ec2.test_artifacts.test_ec2 import test_vllm_on_ec2

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def run_platform_tests(platform: str, images: List[str]):
    """
    Run tests for a specific platform
    """
    instance_type = os.getenv("EC2_GPU_INSTANCE_TYPE")

    if instance_type == "p4d.24xlarge":
        LOGGER.info(f"Skipping tests on {instance_type} instance type")
        return

    if platform == "ec2":
        try:
            ec2_resources = setup()
            print(ec2_resources)
            print("Finished gathering resources required for VLLM EC2 Tests")
            test_vllm_on_ec2(ec2_resources, images[0])
            LOGGER.info("ECS vLLM tests completed successfully")
        except Exception as e:
            LOGGER.error(f"ECS vLLM tests failed: {str(e)}")
            raise
    elif platform == "eks":
        LOGGER.info("Running EKS tests")
        try:
            test_vllm_on_eks(images[0])
            LOGGER.info("EKS vLLM tests completed successfully")
        except Exception as e:
            LOGGER.error(f"EKS vLLM tests failed: {str(e)}")
            raise


def test():
    LOGGER.info("Triggering test from vllm")
    test_type = os.getenv("TEST_TYPE")

    LOGGER.info(f"TEST_TYPE: {test_type}")

    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    dlc_images = os.getenv("DLC_IMAGE") if executor_mode else get_dlc_images()

    ipv6_enabled = os.getenv("ENABLE_IPV6_TESTING", "false").lower() == "true"
    os.environ["ENABLE_IPV6_TESTING"] = "true" if ipv6_enabled else "false"

    commit_id = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", default="unrecognised_commit_id")
    LOGGER.info(f"Commit ID: {commit_id}")

    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    LOGGER.info(f"\nImages URIs:\n{standard_images_list}")

    run_platform_tests(
        platform=test_type,
        images=standard_images_list,
    )


if __name__ == "__main__":
    test()
