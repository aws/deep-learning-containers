import argparse
import logging
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


if __name__ == "__main__":
    main()
