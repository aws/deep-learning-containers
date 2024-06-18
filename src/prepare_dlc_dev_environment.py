import argparse
import logging
import sys
import toml
import re

from config import get_dlc_developer_config_path


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
# LOGGER.addHandler(logging.StreamHandler(sys.stderr))


VALID_TEST_TYPES = [
    "sanity_tests",
    "ec2_tests",
    "ecs_tests",
    "eks_tests",
    "sagemaker_remote_tests",
    "sagemaker_local_tests",
]


VALID_DEV_MODES = ["graviton_mode", "neuronx_mode", "deep_canary_mode"]


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
        "--tests",
        nargs="+",
        choices=VALID_TEST_TYPES,
        default=VALID_TEST_TYPES,
        help="Types of tests to run",
    )
    parser.add_argument(
        "--buildspecs",
        required=True,
        nargs="+",
        help="Path to a buildspec file from the deep-learning-containers folder",
    )

    return parser.parse_args()


class TomlOverrider:
    def __init__(self):
        self._overrides = {"build": {}, "test": {}, "dev": {}, "buildspec_override": {}}
        for dev_mode in VALID_DEV_MODES:
            self._overrides["dev"][dev_mode] = False

    def set_build_frameworks(self, frameworks):
        """
        This method takes a list of frameworks as input and assembles a dictionary with the key
        'build_frameworks' and the value as a list of unique framework names. The resulting
        dictionary is stored in the _overrides attribute of the TomlOverrider object
        """
        if frameworks:
            unique_frameworks = list(dict.fromkeys(frameworks))
            self._overrides["build"]["build_frameworks"] = unique_frameworks

    def set_job_type(self, job_types):
        """
        Job type can be one of (or both) "training" or "inference"

        If job_type is training, set build_training to True, and build_inference to False
        If job type is inference, set build_training to False, and build_inference to True
        If both are set, set both to true
        """
        build_training = "training" in job_types
        build_inference = "inference" in job_types
        self._overrides["build"]["build_training"] = build_training
        self._overrides["build"]["build_inference"] = build_inference

    def set_test_types(self, test_types):
        """
        This method takes a list of test types as input and updates the test overrides dictionary
        based on the provided test types. It assumes that all tests are enabled by default.
        The provided test types will be kept enabled.
        """
        # Disable all tests
        for test_type in VALID_TEST_TYPES:
            self._overrides["test"][test_type] = False

        # Enable the provided test types
        for test_type in test_types:
            self._overrides["test"][test_type] = True

        # Enable all tests if an empty list is provided
        if not test_types:
            for test_type in VALID_TEST_TYPES:
                self._overrides["test"][test_type] = True

    def set_dev_mode(self, dev_mode):
        """
        Set the dev mode based on the user input.
        Valid choices are 'graviton_mode', 'neuronx_mode', and 'deep_canary_mode'.
        """
        # reset all dev modes to False
        for mode in VALID_DEV_MODES:
            self._overrides["dev"][mode] = False
        if isinstance(dev_mode, list):
            raise ValueError("Only one dev mode is allowed at a time.")
        if dev_mode and dev_mode in VALID_DEV_MODES:
            self._overrides["dev"][dev_mode] = True

    def set_buildspec(self, buildspec_paths):
        frameworks = []
        job_types = []
        dev_mode = None

        for buildspec_path in buildspec_paths:
            if not buildspec_path:
                continue

            # Validate the buildspec_path format
            match = self._validate_buildspec_path(buildspec_path)
            if not match:
                continue

            framework, job_type, buildspec_info = self._extract_buildspec_info(match)
            frameworks.append(framework)
            job_types.append(job_type)

            # Check for dev mode and use the first one encountered
            if not dev_mode and buildspec_info:
                dev_mode = self._get_dev_mode(buildspec_info)

            dev_mode_str = f"-{dev_mode.replace('_mode', '')}" if dev_mode else ""
            build_job = f"dlc-pr-{framework}{dev_mode_str}-{job_type}"
            self._overrides["buildspec_override"][build_job] = buildspec_path

        # Reset all dev modes to False and set the first encountered dev mode to True
        self._handle_dev_modes(dev_mode)

        self.set_build_frameworks(frameworks=frameworks)
        self.set_job_type(job_types=job_types)

    def _validate_buildspec_path(self, buildspec_path):
        buildspec_pattern = r"^(\S+)/(training|inference)/buildspec(\S*)\.yml$"
        return re.match(buildspec_pattern, buildspec_path)

    def _extract_buildspec_info(self, match):
        if not match:
            return None, None, None

        framework = match.group(1).replace("/", "_")
        framework_str = f"{framework}-2" if framework == "tensorflow" else framework
        job_type = match.group(2)
        buildspec_info = match.group(3)
        return framework_str, job_type, buildspec_info

    def _get_dev_mode(self, buildspec_info):
        for dm in VALID_DEV_MODES:
            if dm.replace("_mode", "") in buildspec_info:
                return dm
        return None

    def _handle_dev_modes(self, dev_mode):
        if dev_mode is None:
            return

        for mode in VALID_DEV_MODES:
            self._overrides["dev"][mode] = (mode == dev_mode)
    @property
    def overrides(self):
        return self._overrides


def write_toml(toml_path, overrides):
    with open(toml_path, "r") as toml_file_reader:
        loaded_toml = toml.load(toml_file_reader)

    for key, value in overrides.items():
        if key == "buildspec_override":
            for k, v in value.items():
                loaded_toml["buildspec_override"][k] = v
        else:
            for k, v in value.items():
                loaded_toml[key][k] = v

    with open(toml_path, "w") as toml_file_writer:
        output = toml.dumps(loaded_toml).split("\n")
        for line in output:
            toml_file_writer.write(f"{line}\n")


def main():
    args = get_args()
    toml_path = args.partner_toml
    test_types = args.tests
    buildspec_paths = args.buildspecs

    overrider = TomlOverrider()

    # handle frameworks to build
    overrider.set_test_types(test_types=test_types)
    overrider.set_buildspec(buildspec_paths=buildspec_paths)

    LOGGER.info(overrider.overrides)
    write_toml(toml_path, overrides=overrider.overrides)


if __name__ == "__main__":
    main()
