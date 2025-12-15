import os
import sys
import logging
import datetime
import yaml
from typing import Dict, List
from src.buildspec import Buildspec
from test.test_utils import get_buildspec_path
from codebuild_environment import get_cloned_folder_path
from infra.test_infra.validators.platform_validator_utils import get_platform_validator


def create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s PST %(levelname)s %(name)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = lambda *args: datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=-8))
    ).timetuple()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = create_logger(__name__)


def resolve_buildspec_variables(config):
    """Resolve environment variables in buildspec"""
    region = os.getenv("REGION", "us-west-2")
    account_id = os.getenv("ACCOUNT_ID", "")
    config_str = yaml.dump(config)
    config_str = config_str.replace("<set-$REGION-in-environment>", region)
    config_str = config_str.replace("<set-$ACCOUNT_ID-in-environment>", account_id)
    return yaml.safe_load(config_str)


def parse_buildspec(image_uri):
    """Parse buildspec for test configurations"""
    repo_root = get_cloned_folder_path()
    buildspec_path = get_buildspec_path(repo_root)

    if not os.path.exists(buildspec_path):
        raise FileNotFoundError(f"Buildspec file not found: {buildspec_path}")

    BUILDSPEC = Buildspec()
    BUILDSPEC.load(buildspec_path)

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

    return {"tests": tests, "globals": globals_data}


def validate_and_filter_tests(buildspec_data: Dict, test_type: str, base_path: str) -> List[Dict]:
    """Validate and filter tests for the given test type"""
    applicable_tests = []
    validation_errors = []

    all_tests = buildspec_data.get("tests", [])
    LOGGER.info(f"Found {len(all_tests)} total test configurations")

    platform_tests = [test for test in all_tests if test["platform"].startswith(test_type)]
    LOGGER.info(f"Found {len(platform_tests)} applicable test configurations for {test_type}")

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
    """Execute tests for a platform"""
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
