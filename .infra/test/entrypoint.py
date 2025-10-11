import os
import yaml
from typing import Dict, List
from src.config import is_new_test_structure_enabled
from src.buildspec import Buildspec
from .infra.test.ec2.setup import EC2Platform
from .infra.test.eks.setup import EKSPlatform
from .infra.test.validators.platform_validator_utils import get_platform_validator
from test.test_utils import LOGGER, get_dlc_images, get_buildspec_path
from codebuild_environment import get_cloned_folder_path

def resolve_buildspec_variables(config):
    """
    Resolve environment variables in buildspec
    """
    region = os.getenv("REGION", "us-west-2")
    account_id = os.getenv("ACCOUNT_ID", "")

    config_str = yaml.dump(config)
    config_str = config_str.replace("<set-$REGION-in-environment>", region)
    config_str = config_str.replace("<set-$ACCOUNT_ID-in-environment>", account_id)

    return yaml.safe_load(config_str)


def parse_buildspec(image_uri):
    """
    Parse buildspec for test configurations
    """
    repo_root = get_cloned_folder_path()
    buildspec_path = get_buildspec_path(repo_root)

    if not os.path.exists(buildspec_path):
        raise FileNotFoundError(f"Buildspec file not found: {buildspec_path}")

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec_path)

    # Extract test configs
    images = BUILDSPEC.get("images", {})
    image_key = list(images.keys())[0]
    LOGGER.info(f"Using image config: {image_key}")
    image_config = images[image_key]

    tests = image_config.get("tests", [])

    globals_data = {
        "region": BUILDSPEC.get("region"),
        "arch_type": BUILDSPEC.get("arch_type"),
        "framework": BUILDSPEC.get("framework"),
    }

    return {
        "tests": tests,
        "globals": globals_data,
    }


def validate_and_filter_tests(buildspec_data: Dict, test_type: str, base_path: str) -> List[Dict]:
    applicable_tests = []
    validation_errors = []

    # Filter for applicable tests
    all_tests = buildspec_data.get("tests", [])
    LOGGER.info(f"Found {len(all_tests)} total test configurations")

    platform_tests = [test for test in all_tests if test["platform"].startswith(test_type)]
    LOGGER.info(f"Found {len(platform_tests)} applicable test configurations for {test_type}")

    # Validate each applicable test
    for test in platform_tests:
        try:
            validator = get_platform_validator(test["platform"], base_path)
            test_config = {**test, "globals": buildspec_data.get("globals", {})}
            errors = validator.validate(test_config)

            if errors:
                validation_errors.extend(
                    [f"Test {test['platform']}:", *[f"  {error}" for error in errors], ""]
                )
            else:
                applicable_tests.append(test)

        except ValueError as e:
            validation_errors.append(f"Test {test['platform']}: {str(e)}")

    if validation_errors:
        error_msg = "\n".join(validation_errors)
        LOGGER.error("Test validation failed:")
        LOGGER.error(error_msg)
        raise ValueError("Test validation failed. See errors above.")

    LOGGER.info(f"Validated {len(applicable_tests)} test configurations successfully")
    return applicable_tests


def execute_platform_tests(platform, test_config, buildspec_data, image_uri):
    """
    Helper function to execute tests for a platform
    """
    try:
        setup_params = {
            **test_config["params"],
            **buildspec_data["globals"],
            "image_uri": image_uri,
        }

        platform.setup(setup_params)
        LOGGER.info(f"{platform.__class__.__name__} setup completed")

        LOGGER.info(f"Executing {len(test_config['run'])} commands:")
        for cmd in test_config["run"]:
            LOGGER.info(f"  - {cmd}")
            platform.execute_command(cmd)
    except Exception as e:
        LOGGER.error(f"Test failed: {e}")
        raise


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
