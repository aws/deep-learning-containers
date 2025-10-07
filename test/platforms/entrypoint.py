import os
import yaml
from src.config import is_new_test_structure_enabled
from src.buildspec import Buildspec
from test.platforms.infra.ec2.setup import EC2Platform
from test.platforms.infra.eks.setup import EKSPlatform
from test.test_utils import get_framework_from_image_uri
from codebuild_environment import get_cloned_folder_path


def get_buildspec_path():
    """
    Determine buildspec path based on framework and architecture type
    """
    image_uri = os.getenv("DLC_IMAGE", "")
    print(f"Entrypoint - Image URI: {image_uri}")
    framework = get_framework_from_image_uri(image_uri)
    print(f"Entrypoint - Framework: {framework}")

    # Get the repository root directory
    repo_root = get_cloned_folder_path()
    print(f"Repository root: {repo_root}")

    if framework == "vllm":
        if "arm64" in image_uri:
            buildspec_path = os.path.join(repo_root, f"{framework}/buildspec-arm64.yml")
        else:
            buildspec_path = os.path.join(repo_root, f"{framework}/buildspec.yml")
        print(f"Buildspec path: {buildspec_path}")
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


def parse_buildspec():
    """
    Parse buildspec for test configurations
    """
    buildspec_path = get_buildspec_path()
    print(f"Loading buildspec from: {buildspec_path}")
    
    if not os.path.exists(buildspec_path):
        raise FileNotFoundError(f"Buildspec file not found: {buildspec_path}")

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec_path)
    print(f"Buildspec loaded successfully")

    # Extract test configs
    images = BUILDSPEC.get("images", {})
    image_key = list(images.keys())[0]
    print(f"Using image config: {image_key}")
    image_config = images[image_key]

    test_configs = image_config.get("test_configs", {})
    tests = test_configs.get("tests", [])
    print(f"Found {len(tests)} test_configs")

    globals_data = {
        "region": BUILDSPEC.get("region"),
        "arch_type": BUILDSPEC.get("arch_type"),
        "framework": BUILDSPEC.get("framework"),
    }
    print(f"Globals extracted: {globals_data}")

    return {
        "tests": tests,
        "globals": globals_data,
    }


def main():
    print("=== New DLC Test System ===")
    print(f"New test structure enabled: {is_new_test_structure_enabled()}")

    if not is_new_test_structure_enabled():
        print("New test structure not enabled")
        return

    test_type = os.getenv("TEST_TYPE")
    image_uri = os.getenv("DLC_IMAGE")
    
    print(f"Environment variables:")
    print(f"  TEST_TYPE: {test_type}")
    print(f"  DLC_IMAGE: {image_uri}")
    print(f"  REGION: {os.getenv('REGION')}")
    print(f"  ACCOUNT_ID: {os.getenv('ACCOUNT_ID')}")
    
    if not image_uri:
        print("ERROR: DLC_IMAGE environment variable not set")
        return
        
    framework = get_framework_from_image_uri(image_uri)
    print(f"Detected framework: {framework}")

    try:
        buildspec_data = parse_buildspec()
        print(f"Buildspec parsed successfully")
    except Exception as e:
        print(f"ERROR: Failed to parse buildspec: {e}")
        raise

    # Filter for applicable tests
    applicable_tests = [test for test in buildspec_data["tests"] if test["platform"].startswith(test_type)]

    print(f"Found {len(buildspec_data['tests'])} test configurations")
    print(f"Found {len(applicable_tests)} applicable test configurations for {test_type}")
    
    for i, test_config in enumerate(applicable_tests):
        platform_name = test_config["platform"]
        print(f"Test config {i+1}: platform={platform_name}")

        if test_type == "ec2" and platform_name.startswith("ec2"):
            print(f"Executing EC2 test for platform: {platform_name}")
            platform = EC2Platform()
            try:
                setup_params = {**test_config["params"], **buildspec_data["globals"]}
                print(f"Setup parameters: {setup_params}")
                resources = platform.setup(setup_params)
                print(f"Platform setup completed")

                print(f"Executing {len(test_config['run'])} commands:")
                for cmd in test_config["run"]:
                    print(f"  - {cmd}")

            except Exception as e:
                print(f"Test failed: {e}")
                raise
        elif test_type == "eks" and platform_name.startswith("eks"):
            print(f"Executing EKS test for platform: {platform_name}")
            platform = EKSPlatform()
            try:
                setup_params = {**test_config["params"], **buildspec_data["globals"]}
                print(f"Setup parameters: {setup_params}")
                platform.setup(setup_params)
                print(f"Platform setup completed")

                print(f"Executing {len(test_config['run'])} commands:")
                for cmd in test_config["run"]:
                    print(f"  - {cmd}")
                    platform.execute_command(cmd)
            except Exception as e:
                print(f"Test failed: {e}")
                raise
        else:
            print(f"Skipping test config {i+1}: test_type={test_type}, platform={platform_name}")


if __name__ == "__main__":
    main()
