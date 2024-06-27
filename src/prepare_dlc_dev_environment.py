import os
import argparse
import logging
import sys
import re
import requests

import toml
import pprint
import git

from config import get_dlc_developer_config_path
from codebuild_environment import get_cloned_folder_path


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


VALID_TEST_TYPES = [
    "sanity_tests",
    "ec2_tests",
    "ecs_tests",
    "eks_tests",
    "sagemaker_remote_tests",
    "sagemaker_local_tests",
]


VALID_DEV_MODES = ["graviton_mode", "neuronx_mode", "deep_canary_mode"]

DEFAULT_TOML_URL = "https://raw.githubusercontent.com/aws/deep-learning-containers/master/dlc_developer_config.toml"


def restore_default_toml(toml_path):
    """
    Restore the TOML file to its default state from the specified URL
    """
    try:
        response = requests.get(DEFAULT_TOML_URL)
        response.raise_for_status()
        with open(toml_path, "w") as toml_file:
            toml_file.write(response.text)
        LOGGER.info(f"Restored {toml_path} to its default state from {DEFAULT_TOML_URL}")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Error restoring {toml_path}: {e}")


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
        "-t",
        "--tests",
        nargs="+",
        choices=VALID_TEST_TYPES,
        default=VALID_TEST_TYPES,
        help="Types of tests to run",
    )
    parser.add_argument(
        "-b",
        "--buildspecs",
        nargs="+",
        help="Path to a buildspec file from the deep-learning-containers folder",
    )
    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Restore the TOML file to its default state",
    )
    parser.add_argument(
        "-c",
        "--commit",
        action="store_true",
        help="Adds files changed, commits them locally",
    )
    parser.add_argument(
        "-p",
        "--push",
        help="Push change to remote specified (i.e. origin)",
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
        if not test_types:
            return
        # Disable all tests
        for test_type in VALID_TEST_TYPES:
            self._overrides["test"][test_type] = False

        # Enable the provided test types
        for test_type in test_types:
            self._overrides["test"][test_type] = True

    def set_dev_mode(self, dev_mode):
        """
        Set the dev mode based on the user input.
        Valid choices are 'graviton_mode', 'neuronx_mode', and 'deep_canary_mode'.
        """
        # Reset all dev modes to False
        for mode in VALID_DEV_MODES:
            self._overrides["dev"][mode] = False
        if isinstance(dev_mode, list):
            raise ValueError("Only one dev mode is allowed at a time.")
        if dev_mode and dev_mode in VALID_DEV_MODES:
            self._overrides["dev"][dev_mode] = True

    def set_buildspec(self, buildspec_paths):
        """
        This method takes a buildspec path as input and updates the corresponding key in the
        buildspec_override section of the TOML file.
        """
        frameworks = []
        job_types = []
        dev_modes = []

        invalid_paths = []

        # define the expected file path syntax:
        # <framework>/<framework>/<job_type>/buildspec-<version>-<version>.yml
        buildspec_pattern = r"^(\S+)/(training|inference)/buildspec(\S*)\.yml$"

        for buildspec_path in buildspec_paths:
            # validate the buildspec_path format
            match = re.match(buildspec_pattern, buildspec_path)
            if not match or not os.path.exists(
                os.path.join(get_cloned_folder_path(), buildspec_path)
            ):
                LOGGER.warning(
                    f"WARNING! {buildspec_path} does not exist. Moving on to the next one..."
                )
                invalid_paths.append(buildspec_path)
                continue

            # extract the framework, job_type, and version from the buildspec_path
            framework = match.group(1).replace("/", "_")
            frameworks.append(framework)
            framework_str = (
                framework.replace("_", "-") if framework != "tensorflow" else "tensorflow-2"
            )
            job_type = match.group(2)
            job_types.append(job_type)
            buildspec_info = match.group(3)

            dev_mode = None
            for dm in VALID_DEV_MODES:
                if dm.replace("_mode", "") in buildspec_info:
                    dev_mode = dm
                    break
            dev_modes.append(dev_mode)

            # construct the build_job name using the extracted info
            dev_mode_str = f"-{dev_mode.replace('_mode', '')}" if dev_mode else ""
            build_job = f"dlc-pr-{framework_str}{dev_mode_str}-{job_type}"

            self._overrides["buildspec_override"][build_job] = buildspec_path

        if invalid_paths:
            raise RuntimeError(
                f"Found buildspecs that either do not match regex {buildspec_pattern} or do not exist: {invalid_paths}. Please retry, and use tab completion to find valid buildspecs."
            )

        if len(set(dev_modes)) > 1:
            LOGGER.warning(
                f"Only 1 dev mode is allowed, selecting the first mode in the list: {dev_modes[0]}"
            )

        self.set_dev_mode(dev_mode=dev_modes[0])
        self.set_build_frameworks(frameworks=frameworks)
        self.set_job_type(job_types=job_types)

    @property
    def overrides(self):
        return self._overrides


def write_toml(toml_path, overrides):
    unrecognized_options = set()
    with open(toml_path, "r") as toml_file_reader:
        loaded_toml = toml.load(toml_file_reader)

    for key, value in overrides.items():
        for k, v in value.items():
            if loaded_toml.get(key, {}).get(k, None) is None:
                LOGGER.warning(
                    f"WARNING: Writing unrecognized key {key} {k} with value {v} to {toml_path}"
                )
                unrecognized_options.add(k)
            loaded_toml[key][k] = v

    with open(toml_path, "w") as toml_file_writer:
        output = toml.dumps(loaded_toml).split("\n")
        for line in output:
            if line.split("=")[0].strip() in unrecognized_options:
                toml_file_writer.write("# WARNING: Unrecognized key generated below\n")
            toml_file_writer.write(f"{line}\n")


def commit_and_push_changes(changes, remote_push=None, restore=False):
    dlc_repo = git.Repo(os.getcwd(), search_parent_directories=True)
    update_or_restore = "Restore" if restore else "Update"
    commit_message = f"{update_or_restore} {[change.split('deep-learning-containers/')[-1] for change in changes.keys()]}\n"
    for file_name, overrides in changes.items():
        commit_message += f"\n{file_name.split('deep-learning-containers/')[-1]}\n{pprint.pformat(overrides, indent=4)}"
        dlc_repo.git.add(file_name)
    dlc_repo.git.commit("--allow-empty", "-m", commit_message)
    LOGGER.info(f"Committed change\n{commit_message}")

    if remote_push:
        branch = dlc_repo.active_branch.name
        dlc_repo.remotes[remote_push].push(branch)
        LOGGER.info(f"Pushed change to {remote_push}/{branch}")

    return commit_message


def main():
    args = get_args()
    toml_path = args.partner_toml
    test_types = args.tests
    buildspec_paths = args.buildspecs
    restore = args.restore
    to_commit = args.commit
    to_push = args.push

    if not buildspec_paths and not restore:
        LOGGER.error(
            "The developer environment prepare script requires either buildspec or restore options. Please use the '-h' flag to list all options and retry."
        )
        exit(1)
    restore_default_toml(toml_path)
    if restore:
        LOGGER.info(
            f"Restore option found - restoring TOML to its state in {DEFAULT_TOML_URL} and exiting."
        )
        if to_commit:
            commit_and_push_changes(
                {toml_path: f"Restore to {DEFAULT_TOML_URL}"}, remote_push=to_push, restore=True
            )
        return

    overrider = TomlOverrider()

    # handle frameworks to build
    overrider.set_test_types(test_types=test_types)
    overrider.set_buildspec(buildspec_paths=buildspec_paths)

    LOGGER.info(overrider.overrides)
    write_toml(toml_path, overrides=overrider.overrides)
    if to_commit:
        commit_and_push_changes({toml_path: overrider.overrides}, remote_push=to_push)


if __name__ == "__main__":
    main()
