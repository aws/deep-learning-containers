import os
import sys
print(f"current dir: {os.getcwd()}")
print(f"Python path: {sys.path}")
from src.config import is_new_test_structure_enabled
from ec2.setup import EC2Platform
from eks.setup import EKSPlatform
from test_infra_utils import (
    create_logger,
    parse_buildspec,
    validate_and_filter_tests,
    execute_platform_tests,
)
from test.test_utils import get_dlc_images
from codebuild_environment import get_cloned_folder_path

LOGGER = create_logger(__name__)


def main():
    LOGGER.info("================== New DLC Test System ==================")
    LOGGER.info(f"New test structure enabled: {is_new_test_structure_enabled()}")

    test_type = os.getenv("TEST_TYPE")
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    dlc_images = os.getenv("DLC_IMAGE") if executor_mode else get_dlc_images()

    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    LOGGER.info(f"\nImages URIs:\n{standard_images_list}")

    if not standard_images_list:
        raise ValueError("No standard images found")

    image_uri = standard_images_list[0]

    try:
        buildspec_data = parse_buildspec(image_uri)
        LOGGER.info(f"Buildspec parsed successfully")
        framework = buildspec_data["globals"]["framework"]
        LOGGER.info(f"Detected framework: {framework}")
    except Exception as e:
        LOGGER.info(f"ERROR: Failed to parse buildspec: {e}")
        raise

    base_path = get_cloned_folder_path()

    try:
        applicable_tests = validate_and_filter_tests(buildspec_data, test_type, base_path)
        LOGGER.info(f"Found {len(applicable_tests)} valid test configurations for {test_type}")
    except ValueError as e:
        LOGGER.error("Validation failed:")
        LOGGER.error(str(e))
        raise SystemExit(1)

    for i, test_config in enumerate(applicable_tests):
        platform_name = test_config["platform"]
        LOGGER.info(f"Test config {i+1}: platform={platform_name}")

        if test_type == "ec2" and platform_name.startswith("ec2"):
            LOGGER.info(f"Executing EC2 test for platform: {platform_name}")
            execute_platform_tests(EC2Platform(), test_config, buildspec_data, image_uri)
        elif test_type == "eks" and platform_name.startswith("eks"):
            LOGGER.info(f"Executing EKS test for platform: {platform_name}")
            execute_platform_tests(EKSPlatform(), test_config, buildspec_data, image_uri)
        else:
            LOGGER.info(
                f"Skipping test config {i+1}: test_type={test_type}, platform={platform_name}"
            )


if __name__ == "__main__":
    main()
