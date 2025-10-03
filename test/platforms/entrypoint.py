import os
import yaml
from src.config import is_new_test_structure_enabled
from test.platforms.infra.ec2.setup import EC2Platform
from test.test_utils import get_framework_from_image_uri


def get_buildspec_path():
    """
    Determine buildspec path based on framework and architecture type
    """
    image_uri = os.getenv("DLC_IMAGE", "")
    print(f"Entrypoint - Image URI: {image_uri}")
    framework = get_framework_from_image_uri(image_uri)
    print(f"Entrypoint - Framework: {framework}")

    if framework == "vllm":
        if "arm64" in image_uri:
            return f"{framework}/buildspec-arm64.yml"
        else:
            return f"{framework}/buildspec.yml"
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


def parse_buildspec():
    """
    Parse buildspec for test configurations
    """
    buildspec_path = get_buildspec_path()
    print(f"Loading buildspec from: {buildspec_path}")

    with open(buildspec_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Buildspec loaded successfully")

    config = resolve_buildspec_variables(config)
    print(f"Environment variables resolved")

    # extract test configs from buildspec
    images = config.get("images", {})
    image_key = list(images.keys())[0]
    print(f"Using image config: {image_key}")
    image_config = images[image_key]

    tests = image_config.get("test_configs", {}).get("tests", [])
    print(f"Found {len(tests)} test configuration(s)")
    
    globals_data = {
        "region": config.get("region"),
        "arch_type": config.get("arch_type"),
        "framework": config.get("framework"),
    }
    print(f"Globals extracted: {globals_data}")

    return {
        "tests": tests,
        "globals": globals_data,
    }


def main():
    print("=== New DLC Test System ===")

    if not is_new_test_structure_enabled():
        print("New test structure not enabled")
        return

    test_type = os.getenv("TEST_TYPE")
    image_uri = os.getenv("DLC_IMAGE")
    framework = get_framework_from_image_uri(image_uri)

    print(f"Test type: {test_type}")
    print(f"Framework: {framework}")
    print(f"Image: {image_uri}")

    buildspec_data = parse_buildspec()

    for test_config in buildspec_data["tests"]:
        platform_name = test_config["platform"]

        if test_type == "ec2" and platform_name.startswith("ec2"):
            platform = EC2Platform()
            try:
                setup_params = {**test_config["params"], **buildspec_data["globals"]}
                resources = platform.setup(setup_params)

                print(f"Would execute {len(test_config['run'])} commands")
                for cmd in test_config["run"]:
                    print(f"  - {cmd}")

            except Exception as e:
                print(f"Test failed: {e}")
                raise


if __name__ == "__main__":
    main()
