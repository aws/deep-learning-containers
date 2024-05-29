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
        required=True,
        help="ML Framework for which to prepare developer enviornment",
    )
    parser.add_argument(
        "--job_types",
        nargs="+",
        choices=["training", "inference"],
        default=["training", "inference"],
        help="Training and inference containers to prepare developer environment",
    )

    return parser.parse_args()


class TomlOverrider:
    def __init__(self):
        self._overrides = {}

    def set_build_frameworks(self, frameworks):
        """
        Write a function that, given a list of frameworks, assembles a dictionary with key/value pairs:
        {"build_frameworks": ["framework1", "framework2"]}

        Make sure there are no repeats

        frameworks is a list
        return a dictionary object
        """
        unique_frameworks = list(set(frameworks))
        self._overrides["build_frameworks"] = unique_frameworks

    def set_job_type(self, job_types):
        """
        Job type can be one of (or both) "training" or "inference"

        If job_type is training, set build_training to True, and build_inference to False
        If job type is inference, set build_training to False, and build_inference to True
        If both are set, set both to true
        """
        build_training = "training" in job_types
        build_inference = "inference" in job_types
        self._overrides["build_training"] = build_training
        self._overrides["build_inference"] = build_inference

    @property
    def overrides(self):
        return self._overrides


def main():
    args = get_args()
    frameworks = args.frameworks
    job_types = args.job_types

    LOGGER.info(f"Inferring framework to be {frameworks}...")

    overrider = TomlOverrider()

    # Handle frameworks to build
    overrider.set_build_frameworks(frameworks=frameworks)
    overrider.set_job_type(job_types=job_types)

    LOGGER.info(overrider.overrides)

if __name__ == "__main__":
    main()
