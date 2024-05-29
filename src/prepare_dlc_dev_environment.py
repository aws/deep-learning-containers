import argparse
import logging
import sys

from config import get_dlc_developer_config_path


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
# LOGGER.addHandler(logging.StreamHandler(sys.stderr))


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
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["pytorch", "tensorflow"],
        help="ML Framework for which to prepare developer enviornment",
    )

    return parser.parse_args()


def set_build_frameworks(frameworks):
    """
    Write a function that, given a list of frameworks, assembles a dictionary with key/value pairs:
    {"build_frameworks": ["framework1", "framework2"]}

    Make sure there are no repeats

    frameworks is a list
    return a dictionary object
    """
    unique_frameworks = list(set(frameworks))
    return {"build_frameworks": unique_frameworks}
    #pass


def main():
    args = get_args()
    frameworks = args.frameworks
    LOGGER.info(f"Inferring framework to be {frameworks}...")
    LOGGER.info(set_build_frameworks(frameworks))


if __name__ == "__main__":
    main()
