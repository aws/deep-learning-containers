import os
import yaml
from src.config import is_new_test_structure_enabled
from src.buildspec import Buildspec
from test.platforms.infra.ec2.setup import EC2Platform
from test.platforms.infra.eks.setup import EKSPlatform
from test.test_utils import LOGGER, get_framework_from_image_uri, get_dlc_images
from codebuild_environment import get_cloned_folder_path


def get_buildspec_path(image_uri):
    """
    Determine buildspec path based on framework and architecture type
    """
    LOGGER.info(f"Entrypoint - Image URI: {image_uri}")
    framework = get_framework_from_image_uri(image_uri)
    LOGGER.info(f"Entrypoint - Framework: {framework}")
    repo_root = get_cloned_folder_path()

    if framework == "vllm":
        if "arm64" in image_uri:
            buildspec_path = os.path.join(repo_root, f"{framework}/buildspec-arm64.yml")
        else:
            buildspec_path = os.path.join(repo_root, f"{framework}/buildspec.yml")
        return buildspec_path
    # other frameworks do not have test_configs yet
    else:
        raise NotImplementedError(f"Buildspec parsing not yet implemented for {framework}")


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
    buildspec_path = get_buildspec_path(image_uri)

    if not os.path.exists(buildspec_path):
        raise FileNotFoundError(f"Buildspec file not found: {buildspec_path}")

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec_path)

    # Extract test configs
    images = BUILDSPEC.get("images", {})
    image_key = list(images.keys())[0]
    LOGGER.info(f"Using image config: {image_key}")
    image_config = images[image_key]

    test_configs = image_config.get("test_configs", {})
    tests = test_configs.get("tests", [])

    globals_data = {
        "region": BUILDSPEC.get("region"),
        "arch_type": BUILDSPEC.get("arch_type"),
        "framework": BUILDSPEC.get("framework"),
    }

    return {
        "tests": tests,
        "globals": globals_data,
    }


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
        LOGGER.error("No standard images found")
        raise ValueError("No standard images found")

    image_uri = standard_images_list[0]
    framework = get_framework_from_image_uri(image_uri)
    LOGGER.info(f"Detected framework: {framework}")

    try:
        buildspec_data = parse_buildspec(image_uri)
        LOGGER.info(f"Buildspec parsed successfully")
    except Exception as e:
        LOGGER.info(f"ERROR: Failed to parse buildspec: {e}")
        raise

    # Filter for applicable tests
    applicable_tests = [
        test for test in buildspec_data["tests"] if test["platform"].startswith(test_type)
    ]

    LOGGER.info(f"Found {len(buildspec_data['tests'])} test configurations")
    LOGGER.info(f"Found {len(applicable_tests)} applicable test configurations for {test_type}")

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
