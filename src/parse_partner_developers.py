import argparse
import logging
import os
import sys

from config import get_dlc_developer_config_path, parse_dlc_developer_configs


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def get_args():
    """
    Manage arguments to this script when called directly
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partner_toml",
        default=get_dlc_developer_config_path(),
        help="TOML file with partner developer information",
    )
    return parser.parse_args()


def main():
    args = get_args()
    partner_dev = parse_dlc_developer_configs("dev", "partner_developer", tomlfile=args.partner_toml)

    if partner_dev:
        LOGGER.info(f"PARTNER_DEVELOPER: {partner_dev.upper()}")
        LOGGER.info(f"PR_NUMBER: {os.getenv('PR_NUMBER', os.getenv('CODEBUILD_SOURCE_VERSION', '')).replace('/', '-')}")
        LOGGER.info(f"COMMIT_ID: {os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}")
        test_trigger = os.getenv("TEST_TRIGGER")
        if test_trigger:
            LOGGER.info(f"TEST_TRIGGER: {test_trigger}")


if __name__ == "__main__":
    main()
