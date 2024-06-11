import argparse
import logging
import sys
import toml

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
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=[
            "benchmark",
            "ec2",
            "ecs",
            "eks",
            "sagemaker_remote",
            "sagemaker_local",
        ],
        default=["ec2", "ecs", "eks", "sagemaker_remote", "sagemaker_local"],
        help="Types of tests to run",
    )
    parser.add_argument(
        "--dev_mode",
        choices=["graviton", "neuron", "deep_canary"],
        default=None,
        help="Enable developer mode for specific hardware targets",
    )

    return parser.parse_args()


class TomlOverrider:
    def __init__(self):
        self._overrides = {"build": {}}

    def set_build_frameworks(self, frameworks):
        """
        This method takes a list of frameworks as input and assembles a dictionary with the key
        'build_frameworks' and the value as a list of unique framework names. The resulting
        dictionary is stored in the _overrides attribute of the TomlOverrider object
        """
        unique_frameworks = list(set(frameworks))
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
        This method takes a list of test types as input and generates the test overrides dictionary
        based on the provided test types. It focuses on the main five test cases: ec2, ecs, eks,
        sagemaker_remote, sagemaker_local. If no test types are provided, it sets all tests except
        benchmark to True by default. The benchmark test is intentionally disabled by default in
        this case.
        """
        if not test_types:
            # If the test_types list is empty (no test types provided)
            # Set all tests except benchmark to True by default
            self._overrides["test"] = {
                "sanity_tests": True,  # sanity_tests are always True
                "safety_check_test": False,
                "ecr_scan_allowlist_feature": False,
                "ecs_tests": True,  # ecs_tests set to True by default
                "eks_tests": True,  # eks_tests set to True by default
                "ec2_tests": True,  # ec2_tests set to True by default
                "ec2_benchmark_tests": False,  # ec2_benchmark_tests set to False by default
                "ec2_tests_on_heavy_instances": False,
                "sagemaker_local_tests": True,  # sagemaker_local_tests set to True by default
                "sagemaker_remote_tests": True,  # sagemaker_remote_tests set to True by default
                "sagemaker_efa_tests": False,
                "sagemaker_rc_tests": False,
                "sagemaker_benchmark_tests": False,
                "nightly_pr_test_mode": False,
                "use_scheduler": False,
            }
        else:
            # If the test_types list is not empty (test types are provided)
            self._overrides["test"] = {
                "sanity_tests": True,  # sanity_tests are always True
                "safety_check_test": False,
                "ecr_scan_allowlist_feature": False,
                "ecs_tests": "ecs" in test_types,  # ecs_tests set to True if "ecs" is in test_types
                "eks_tests": "eks" in test_types,  # eks_tests set to True if "eks" is in test_types
                "ec2_tests": "ec2" in test_types,  # ec2_tests set to True if "ec2" is in test_types
                "ec2_benchmark_tests": "benchmark" in test_types,  # ec2_benchmark_tests set to True if "benchmark" is in test_types
                "ec2_tests_on_heavy_instances": False,
                "sagemaker_local_tests": "sagemaker_local" in test_types,  # sagemaker_local_tests set to True if "sagemaker_local" is in test_types
                "sagemaker_remote_tests": "sagemaker_remote" in test_types,  # sagemaker_remote_tests set to True if "sagemaker_remote" is in test_types
                "sagemaker_efa_tests": False,
                "sagemaker_rc_tests": False,
                "sagemaker_benchmark_tests": False,
                "nightly_pr_test_mode": False,
                "use_scheduler": False,
            }

    def set_dev_mode(self, dev_mode):
        """
        Set the dev mode based on the user input.
        Valid choices are 'graviton', 'neuron', and 'deep_canary'.
        """
        self._overrides["dev"]["graviton_mode"] = False
        self._overrides["dev"]["neuron_mode"] = False
        self._overrides["dev"]["deep_canary_mode"] = False

        mode_mapping = {
            "graviton": "graviton_mode",
            "neuron": "neuron_mode",
            "deep_canary": "deep_canary_mode",
        }

        if dev_mode in mode_mapping:
            self._overrides["dev"][mode_mapping[dev_mode]] = True

    @property
    def overrides(self):
        return self._overrides


def write_toml(toml_path, overrides):
    with open(toml_path, "r") as toml_file_reader:
        loaded_toml = toml.load(toml_file_reader)
    for key, value in overrides.items():
        for k, v in value.items():
            loaded_toml[key][k] = v
    with open(toml_path, "w") as toml_file_writer:
        output = toml.dumps(loaded_toml).split("\n")
        for line in output:
            toml_file_writer.write(f"{line}\n")


def main():
    args = get_args()
    frameworks = args.frameworks
    job_types = args.job_types
    toml_path = args.partner_toml
    test_types = args.tests
    dev_mode = args.dev_mode

    LOGGER.info(f"Inferring framework to be {frameworks}...")

    overrider = TomlOverrider()

    # Handle frameworks to build
    overrider.set_build_frameworks(frameworks=frameworks)
    overrider.set_job_type(job_types=job_types)
    overrider.set_test_types(test_types=test_types)
    overrider.set_dev_mode(dev_mode=dev_mode)

    LOGGER.info(overrider.overrides)
    write_toml(toml_path, overrides=overrider.overrides)


if __name__ == "__main__":
    main()
