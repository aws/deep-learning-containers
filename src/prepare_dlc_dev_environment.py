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
            "sanity_tests",
            "ec2_tests",
            "ecs_tests",
            "eks_tests",
            "sagemaker_remote_tests",
            "sagemaker_local_tests",
        ],
        default=[
            "sanity_tests",
            "ec2_tests",
            "ecs_tests",
            "eks_tests",
            "sagemaker_remote_tests",
            "sagemaker_local_tests",
        ],
        help="Types of tests to run",
    )
    parser.add_argument(
        "--dev_mode",
        choices=["graviton_mode", "neuronx_mode", "deep_canary_mode"],
        default=None,
        help="Enable developer mode for specific hardware targets",
    )
    parser.add_argument(
        "--buildspec",
        help="Path to a buildspec file from the deep-learning-containers folder",
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
        This method takes a list of test types as input and updates the test overrides dictionary
        based on the provided test types. It assumes that all tests are enabled by default, except
        for ec2_benchmark_tests. The provided test types will be kept enabled.
        """

        # Enable the provided test types and the always_enabled_tests
        for test_type in test_types:
            self._overrides["test"][test_type] = True

        # Disable ec2_benchmark_tests by default
        #self._overrides["test"]["ec2_benchmark_tests"] = False
        #global variable (list) store choices, refernece global v here and if not there then se
        #2: set false then iterate to find True

    def set_dev_mode(self, dev_mode):
        """
        Set the dev mode based on the user input.
        Valid choices are 'graviton_mode', 'neuron_mode', and 'deep_canary_mode'.
        """
        if dev_mode:
            self._overrides["dev"][dev_mode] = True

    def set_buildspec(self, buildspec_path):
        """
        This method takes a buildspec path as input and updates the corresponding key in the
        buildspec_override section of the TOML file.
        """
        if not buildspec_path:
            return

        # Infer the build job from the buildspec path
        # Example: "habana/tensorflow/training/buildspec-2-10.yml" -> "dlc-pr-tensorflow-2-habana-training"
        parts = buildspec_path.split("/")
        framework = parts[0]
        job_type = parts[2]
        build_job = f"dlc-pr-{framework}-{job_type}"

        # Update the corresponding key in the buildspec_override section
        self._overrides["buildspec_override"][build_job] = buildspec_path


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
    frameworks = args.frameworks
    job_types = args.job_types
    toml_path = args.partner_toml
    test_types = args.tests
    dev_mode = args.dev_mode
    buildspec_path = args.buildspec

    LOGGER.info(f"Inferring framework to be {frameworks}...")

    overrider = TomlOverrider()

    # handle frameworks to build
    overrider.set_build_frameworks(frameworks=frameworks)
    overrider.set_job_type(job_types=job_types)
    overrider.set_test_types(test_types=test_types)
    overrider.set_dev_mode(dev_mode=dev_mode)
    overrider.set_buildspec(buildspec_path)

    LOGGER.info(overrider.overrides)
    write_toml(toml_path, overrides=overrider.overrides)


if __name__ == "__main__":
    main()
