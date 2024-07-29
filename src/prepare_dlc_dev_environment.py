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
from packaging.version import Version
from pathlib import Path


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
        exit(1)


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
    parser.add_argument(
        "-n",
        action="store_true",
        help="Create a currency buildspec and update toml",
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


def extract_path_components(path, pattern):
    """
    Extract framework, job_type, major_version, minor_version, and other components
    from the provided path using the given regular expression pattern.
    """
    match = re.match(pattern, path)
    if not match:
        raise ValueError(f"Invalid path format: {path}")
    return match.groups()


def find_latest_version_path(framework, job_type, optional_tag, major_version, extra_tag):
    """
    Find the path to the latest existing version of the buildspec file based on the provided components.
    Special condition checks if file is a graviton file
    """
    path_prefix = os.path.join(get_cloned_folder_path(), framework, job_type)
    graviton_pattern = r"buildspec-graviton-(\d+)-(\d+)(?:-{})?\.yml".format(extra_tag or r"\w*")
    non_graviton_pattern = r"buildspec(?:-{})?-(\d+)-(\d+)(?:-{})?\.yml".format(
        optional_tag or r"\w*", extra_tag or r"\w*"
    )
    latest_version = (int(major_version), 0)
    latest_path = None

    for file_name in os.listdir(path_prefix):
        graviton_match = re.match(graviton_pattern, file_name)
        non_graviton_match = re.match(non_graviton_pattern, file_name)

        if graviton_match:
            major_version_str, minor_version_str = graviton_match.groups()[:2]
            version = (int(major_version_str), int(minor_version_str))
            if version > latest_version:
                latest_version = version
                latest_path = os.path.join(path_prefix, file_name)
        elif non_graviton_match:
            major_version_str, minor_version_str = non_graviton_match.groups()[:2]
            minor_version_str = int(minor_version_str)
            if extra_tag:
                version = (int(major_version_str), minor_version_str, extra_tag)
            else:
                version = (int(major_version_str), minor_version_str)
            if version > latest_version:
                latest_version = version
                latest_path = os.path.join(path_prefix, file_name)

    return latest_path


def generate_new_file_content(
    previous_version_path, major_version, minor_version, optional_tag, extra_tag
):
    """
    Generate the content for the new buildspec file with the updated version, short_version, and build_tag_override values.
    """
    new_version = f"{major_version}.{minor_version}.0"
    with open(previous_version_path, "r") as prev_file:
        content = prev_file.readlines()
        for i, line in enumerate(content):
            if line.startswith("version: &VERSION "):
                content[i] = f"version: &VERSION {new_version}\n"
            elif line.startswith("short_version: &SHORT_VERSION "):
                content[i] = f'short_version: &SHORT_VERSION "{major_version}.{minor_version}"\n'
            elif line.strip().startswith("autopatch_build"):
                content[i] = f"# {line}"
            elif line.strip().startswith("# build_tag_override:"):
                build_tag_parts = line.strip().split('"')
                build_tag_handle, old_version_and_rest = build_tag_parts[1].split(":", 1)
                build_tag_rest = (
                    old_version_and_rest.split("-", 1)[1] if "-" in old_version_and_rest else ""
                )
                new_build_tag_override = f'"{build_tag_handle}:{new_version}-{build_tag_rest}"'
                content[i] = f"    # build_tag_override: {new_build_tag_override}\n"

    return content


def create_new_file_with_updated_version(currency_path, updated_content, previous_version_path):
    """
    Create a new buildspec file with the updated content and update the pointer file.
    """
    new_file_path = os.path.join(get_cloned_folder_path(), currency_path)
    if os.path.exists(new_file_path):
        LOGGER.error(f"ERROR: File {new_file_path} already exists, please enter a new file")
        exit(1)

    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, "w") as new_file:
        new_file.writelines(updated_content)

    LOGGER.info(f"Created {new_file_path} using {previous_version_path}")

    # Update the graviton pointer file
    graviton_pointer_file_path = os.path.join(
        os.path.dirname(os.path.dirname(new_file_path)), "inference", "buildspec-graviton.yml"
    )

    if "graviton" in new_file_path and os.path.exists(graviton_pointer_file_path):
        update_pointer_file(graviton_pointer_file_path, currency_path)
    elif "graviton" in new_file_path:
        LOGGER.warning(f"Graviton pointer file not found at {graviton_pointer_file_path}")
    else:
        pointer_file_path = os.path.join(os.path.dirname(new_file_path), "buildspec.yml")
        if os.path.exists(pointer_file_path):
            update_pointer_file(pointer_file_path, currency_path)
        else:
            LOGGER.warning(f"Pointer file not found at {pointer_file_path}")


def update_pointer_file(pointer_file_path, new_buildspec_path):
    """
    Update the pointer file with the path to the newly created buildspec file.
    """
    with open(pointer_file_path, "r") as pointer_file:
        content = pointer_file.readlines()

    for i, line in enumerate(content):
        if line.startswith("buildspec_pointer:"):
            # Remove the path prefix for all pointer files
            new_buildspec_path = os.path.basename(new_buildspec_path)
            content[i] = f"buildspec_pointer: {new_buildspec_path}\n"
            break

    with open(pointer_file_path, "w") as pointer_file:
        pointer_file.writelines(content)

    LOGGER.info(f"Updated pointer file at {pointer_file_path}")


def handle_currency_option(currency_paths):
    """
    Handle the --new_currency option by creating new buildspec files with incremented minor versions
    and updating the TOML file with the information from the new buildspec file paths.
    """
    buildspec_pattern = (
        r"^(\w+)/(training|inference)/buildspec(?:-(\w+))?-(\d+)-(\d+)(?:-(.+))?\.yml$"
    )
    for currency_path in currency_paths:
        if not validate_currency_path(currency_path):
            continue
        (
            framework,
            job_type,
            optional_tag,
            major_version,
            minor_version,
            extra_tag,
        ) = extract_path_components(currency_path, buildspec_pattern)

        latest_version_path = find_latest_version_path(
            framework, job_type, optional_tag, major_version, extra_tag
        )
        if latest_version_path:
            updated_content = generate_new_file_content(
                latest_version_path, major_version, minor_version, optional_tag, extra_tag
            )
            create_new_file_with_updated_version(
                currency_path, updated_content, latest_version_path
            )
        else:
            LOGGER.warning(f"No previous version found for {currency_path}")


def validate_currency_path(currency_path):
    """
    Validate the currency path format using the provided regular expression pattern.
    """
    buildspec_pattern = (
        r"^(?:(\w+)/)?(\w+)/(training|inference)/buildspec(?:-(\w+))?-(\d+)-(\d+)(?:-(.+))?\.yml$"
    )
    match = re.match(buildspec_pattern, currency_path)
    if match:
        (
            extra_framework,
            framework,
            job_type,
            optional_tag,
            major_version,
            minor_version,
            extra_tag,
        ) = match.groups()
        if extra_framework:
            LOGGER.error(
                f"ERROR: {extra_framework} is not currently supported for currency feature. Please provide a supported framework (pytorch/tensorflow)."
            )
            exit(1)
    else:
        raise ValueError(
            f"Invalid currency path format: {currency_path}. "
            f"\nExpected format: <framework>/<job_type>/buildspec[-<optional_tag>]-<major_version>-<minor_version>[-<extra_tag>].yml\n"
            f"For example: pytorch/inference/buildspec-2-3-ec2.yml or pytorch/inference/buildspec-graviton-2-3.yml"
        )
    return True


def create_dockerfile_paths(buildspec_paths):
    for buildspec_path in buildspec_paths:
        buildspec_dir = os.path.dirname(buildspec_path)
        with open(os.path.join(get_cloned_folder_path(), buildspec_path), "r") as buildspec_file:
            buildspec_content = buildspec_file.read()

        # Extract placeholders from buildspec content
        short_version_match = re.search(
            r"short_version: &SHORT_VERSION \"([\d\.]+)\"", buildspec_content
        )
        short_version = short_version_match.group(1) if short_version_match else ""

        python_version_match = re.search(
            r"python_version: &DOCKER_PYTHON_VERSION (\w+)", buildspec_content
        )
        python_version = python_version_match.group(1) if python_version_match else ""

        device_type_match = re.search(r"device_type: &DEVICE_TYPE (\w+)", buildspec_content)
        device_type = device_type_match.group(1) if device_type_match else ""

        cuda_version_match = re.search(r"cuda_version: &CUDA_VERSION (\w+)", buildspec_content)
        cuda_version = cuda_version_match.group(1) if cuda_version_match else ""

        for line in buildspec_content.splitlines():
            if line.strip().startswith("docker_file:"):
                docker_file_path = evaluate_docker_file_path(
                    line, buildspec_dir, short_version, python_version, cuda_version, device_type
                )
                create_docker_file(docker_file_path)


def evaluate_docker_file_path(
    docker_file_line, buildspec_dir, short_version, python_version, cuda_version, device_type
):
    docker_file_parts = docker_file_line.strip().split(":")[1].strip()
    docker_file_path = os.path.join(
        buildspec_dir, *re.split(r"!join\s*\[\s*", docker_file_parts)[1:]
    )
    docker_file_path = docker_file_path.replace("*SHORT_VERSION", short_version)
    docker_file_path = docker_file_path.replace("*DOCKER_PYTHON_VERSION", python_version)
    docker_file_path = docker_file_path.replace("*CUDA_VERSION", cuda_version)
    docker_file_path = docker_file_path.replace("*DEVICE_TYPE", device_type)
    docker_file_path = re.sub(r"[, ]+", "", docker_file_path)
    docker_file_path = docker_file_path.rstrip("]")

    # Handle different formats
    if docker_file_path.endswith("/Dockerfile."):
        docker_file_path = docker_file_path + device_type
    elif docker_file_path.endswith("/Dockerfile.,"):
        docker_file_path = docker_file_path[:-1] + f".{device_type}"
    return Path(docker_file_path)


def create_docker_file(docker_file_path):
    docker_file_dir = os.path.dirname(docker_file_path)
    os.makedirs(docker_file_dir, exist_ok=True)
    try:
        with open(docker_file_path, "w") as docker_file:
            docker_file.write("")
        LOGGER.info(f"Created {docker_file_path}")
    except Exception as e:
        LOGGER.warning(f"WARNING: Failed to create {docker_file_path}. Error: {e}")


def main():
    args = get_args()
    toml_path = args.partner_toml
    test_types = args.tests
    buildspec_paths = args.buildspecs
    restore = args.restore
    to_commit = args.commit
    to_push = args.push
    is_currency = args.n

    # Update to require 1 of 3 options
    if not buildspec_paths and not restore:
        LOGGER.error("No options provided. Please use the '-h' flag to list all options and retry.")
        exit(1)

    restore_default_toml(toml_path)

    # In case of -rc used
    if restore:
        LOGGER.debug(
            f"Restore option found - restoring TOML to its state in {DEFAULT_TOML_URL} and exiting."
        )
        if to_commit:
            commit_and_push_changes(
                {toml_path: f"Restore to {DEFAULT_TOML_URL}"}, remote_push=to_push, restore=True
            )
        return

    overrider = TomlOverrider()

    # Handle the -n option
    if is_currency:
        handle_currency_option(buildspec_paths)
        create_dockerfile_paths(buildspec_paths)

    # handle frameworks to build
    if buildspec_paths:
        overrider.set_test_types(test_types=test_types)
        overrider.set_buildspec(buildspec_paths=buildspec_paths)

    LOGGER.info(overrider.overrides)
    write_toml(toml_path, overrides=overrider.overrides)
    if to_commit:
        commit_and_push_changes({toml_path: overrider.overrides}, remote_push=to_push)


if __name__ == "__main__":
    main()
