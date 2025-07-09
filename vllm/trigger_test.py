import os
import logging

from test.test_utils import get_dlc_images

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def main():

    print("Triggering test from vllm")
    test_type = os.getenv("TEST_TYPE")

    print("TEST_TYPE", test_type)
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    dlc_images = os.getenv("DLC_IMAGE") if executor_mode else get_dlc_images()
    # Enable IPv6 testing from environment variable
    ipv6_enabled = os.getenv("ENABLE_IPV6_TESTING", "false").lower() == "true"
    os.environ["ENABLE_IPV6_TESTING"] = "true" if ipv6_enabled else "false"
    # Executing locally ona can provide commit_id or may ommit it. Assigning default value for local executions:
    commit_id = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", default="unrecognised_commit_id")

    print("Commit ID", commit_id)

    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    print("\n Images URIs \n", standard_images_list)
