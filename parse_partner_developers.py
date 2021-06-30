import argparse
import logging
import sys

import toml


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


PARTNER_TOML = "dlc_developer_config.toml"


def get_args():
    """
    Manage arguments to this script when called directly
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partner_toml",
        default=PARTNER_TOML,
        help="TOML file with partner developer information",
    )
    return parser.parse_args()


def parse_partner_toml(partner_toml):
    partners = toml.load(partner_toml)
    partner_dev = partners.get("dev", {}).get("partner_developer")
    if partner_dev:
        LOGGER.info(f"PARTNER_DEVELOPER: {partner_dev.upper()}")


def main():
    args = get_args()
    parse_partner_toml(args.partner_toml)


if __name__ == "__main__":
    main()
