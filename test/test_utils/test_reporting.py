import csv
import datetime
import os
import re

from invoke.context import Context

from test.test_utils import LOGGER
from test.test_utils.ec2 import get_instance_num_gpus


def get_test_coverage_file_path():
    cwd = os.getcwd()

    dlc_dir = cwd.split('/test/')[0]

    return os.path.join(
        dlc_dir,
        "test",
        f"test_coverage_report-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv",
    )


def get_marker_arg_value(item_obj, marker_name, default=None):
    """
    Function to return the argument value of a pytest marker -- if it does not exist, fall back to a default.
    If the default does not exist and the option does not exist, raise an error.
    :param item_obj: pytest item object
    :param marker_name: name of the pytest marker
    :param default: default return value -- if None, assume this is a required marker
    :return: First arg value for the marker or the default value
    """
    markers = [mark for mark in item_obj.iter_markers(name=marker_name)]
    if not markers:
        if not default:
            raise RequiredMarkerNotFound(f"PyTest Marker {marker_name} is required on function {item_obj.name}")
        return default
    else:
        return markers[0].args[0]


class TestReportGenerator:
    """
    Class to generate test report files
    """

    ALLOWED_SINGLE_GPU_TESTS = ("telemetry", "test_framework_version_gpu")
    SM_REPOS = (
        "pytorch-training",
        "pytorch-inference",
        "tensorflow-tensorflow1_training",
        "tensorflow-tensorflow2_training",
        "mxnet-training",
        "mxnet-inference",
    )
    COVERAGE_DOC_EXECUTABLE = "pytest -s --collect-only --generate-coverage-doc"

    def __init__(self, items, test_coverage_file=None, is_sagemaker=False):
        """
        The TestReportGenerator requires a list of pytest items. The "generate_coverage_doc" method can be
        used for most cases

        :param items: list of pytest item objects
        :param test_coverage_file: optional -- specify the name of the coverage file to write to
        """
        self.items = items
        self.test_coverage_file = get_test_coverage_file_path() if not test_coverage_file else test_coverage_file
        self.is_sagemaker = is_sagemaker
        self.failure_conditions = {}

    def handle_single_gpu_instances_test_report(self, function_key, function_keywords, processor="gpu"):
        """
        Generally, we do not want tests running on single gpu instances. However, there are exceptions to this rule.
        This method is used to determine whether we need to raise an error with report generation or not, based on
        whether we are using single gpu instances or not in a given test function.

        :param function_key: local/path/to/function::function_name
        :param function_keywords: string of keywords associated with the test function
        :param processor: whether the test is for cpu, gpu or both
        :return: processor if not single gpu instance, else "single_gpu", and a dict with updated failure messages
        """

        # Define conditions where we allow a test function to run with a single gpu instance
        whitelist_single_gpu = False
        allowed_single_gpu = self.ALLOWED_SINGLE_GPU_TESTS

        # Regex in order to determine the gpu instance type
        gpu_instance_pattern = re.compile(r"\w+\.\d*xlarge")
        gpu_match = gpu_instance_pattern.search(function_keywords)

        if gpu_match:
            instance_type = gpu_match.group()
            num_gpus = get_instance_num_gpus(instance_type=instance_type)

            for test in allowed_single_gpu:
                if test in function_key:
                    whitelist_single_gpu = True
                    break
            if num_gpus == 1:
                processor = "single_gpu"
                if not whitelist_single_gpu:
                    single_gpu_failure_message = (
                        f"Function uses single-gpu instance type {instance_type}. "
                        f"Please use multi-gpu instance type."
                    )
                    if not self.failure_conditions.get(function_key):
                        self.failure_conditions[function_key] = [single_gpu_failure_message]
                    else:
                        self.failure_conditions[function_key].append(single_gpu_failure_message)

        return processor

    def assemble_report_failure_message(self):
        """
        Assemble the failure message if there are any to raise

        :return: the final failure message string
        """
        final_message = ""
        total_issues = 0
        for func, messages in self.failure_conditions.items():
            final_message += f"******Problems with {func}:******\n"
            for idx, message in enumerate(messages):
                final_message += f"{idx+1}. {message}\n"
                total_issues += 1
        final_message += f"TOTAL ISSUES: {total_issues}"

        return final_message, total_issues

    def write_test_coverage_file(self, test_coverage_info):
        """
        Write out the test coverage file based on a dictionary defining key/value pairs of test coverage
        information

        :param test_coverage_info: dict representing the test coverage information
        """
        # Assemble the list of headers from one item in the dictionary
        field_names = []
        for _key, header in test_coverage_info.items():
            for field_name, _value in header.items():
                field_names.append(field_name)
            break

        # Write to the test coverage file
        file_open_type = "w+"
        if self.is_sagemaker and os.path.exists(self.test_coverage_file):
            file_open_type = "a"
        with open(self.test_coverage_file, file_open_type) as tc_file:
            writer = csv.DictWriter(tc_file, delimiter=",", fieldnames=field_names)
            if file_open_type == "w+":
                writer.writeheader()

            for _func_key, info in test_coverage_info.items():
                writer.writerow(info)

    @staticmethod
    def _infer_field_value(default, options, *comparison_str):
        """
        For a given test coverage report field, determine the value based on whether the options are in keywords or
        file paths.

        :param default: default return value if the field is not found
        :param options: tuple of possible options -- i.e. ("training", "inference")
        :param comparison_str: keyword string, filepath string
        :return: field value <str>
        """
        for comp in comparison_str:
            for option in options:
                if option in comp:
                    return option.strip("_")
        return default

    def generate_sagemaker_reports(self):
        """
        Append SageMaker data to the report
        """
        ctx = Context()
        git_repo_path = os.getcwd().split('/test/')[0]

        for repo in self.SM_REPOS:
            framework, job_type = repo.split("-")
            with ctx.cd(os.path.join(git_repo_path, "test", "sagemaker_tests", framework, job_type)):
                # We need to install requirements in order to use the SM pytest frameworks
                ctx.run("pip install -r requirements.txt", warn=True)
                ctx.run(f"{self.COVERAGE_DOC_EXECUTABLE} integration/")

        # Handle TF inference remote tests
        tf_inf_path = os.path.join(
            git_repo_path, "test", "sagemaker_tests", "tensorflow", "inference")

        with ctx.cd(tf_inf_path):
            # Install TF inference pip requirements
            ctx.run("pip install -r requirements.txt", warn=True)
            with ctx.cd(os.path.join(tf_inf_path, "test", "integration")):
                # Handle local tests
                ctx.run(f"{self.COVERAGE_DOC_EXECUTABLE} --framework-version 2 local/")

                # Handle remote integration tests
                ctx.run(f"{self.COVERAGE_DOC_EXECUTABLE} sagemaker/")

    def generate_coverage_doc(self, framework=None, job_type=None):
        """
        Generate the test coverage docs based on pytest item objects

        :param framework: str, ML framework
        :param job_type: str, training or inference
        """
        test_cov = {}
        for item in self.items:
            # Define additional csv options
            function_name = item.name.split("[")[0]
            function_key = f"{item.fspath}::{function_name}"
            str_fspath = str(item.fspath)
            str_keywords = str(item.keywords)

            # Construct Category and Github_Link fields based on the filepath
            category = str_fspath.split("/dlc_tests/")[-1].split("/")[0]
            if self.is_sagemaker:
                category = "sagemaker_local" if "local" in str_fspath else "sagemaker"
            github_link = (
                f"https://github.com/aws/deep-learning-containers/blob/master/"
                f"{str_fspath.split('/deep-learning-containers/')[-1]}"
            )

            # Only create a new test coverage item if we have not seen the function before. This is a necessary step,
            # as parametrization can make it appear as if the same test function is a unique test function
            if test_cov.get(function_key):
                continue

            # Based on keywords and filepaths, assign values
            framework_scope = (
                framework
                if framework
                else self._infer_field_value("all", ("mxnet", "tensorflow", "pytorch"), str_fspath)
            )
            job_type_scope = (
                job_type
                if job_type
                else self._infer_field_value("both", ("training", "inference"), str_fspath, str_keywords)
            )
            integration_scope = self._infer_field_value(
                "general integration",
                ("_dgl_", "smdebug", "gluonnlp", "smexperiments", "_mme_", "pipemode", "tensorboard", "_s3_", "nccl"),
                str_keywords,
            )
            model_scope = self._infer_field_value(
                "N/A", ("mnist", "densenet", "squeezenet", "half_plus_two", "half_plus_three"), str_keywords
            )
            num_instances = self._infer_field_value(
                1, ("_multinode_", "_multi-node_", "_multi_node_", "_dist_"), str_fspath, str_keywords
            )
            if num_instances != 1:
                num_instances = "multinode"
            processor_scope = self._infer_field_value("all", ("cpu", "gpu", "eia"), str_keywords)
            if processor_scope == "gpu":
                processor_scope = self.handle_single_gpu_instances_test_report(function_key, str_keywords)

            # Create a new test coverage item if we have not seen the function before. This is a necessary step,
            # as parametrization can make it appear as if the same test function is a unique test function
            test_cov[function_key] = {
                "Category": category,
                "Name": function_name,
                "Scope": framework_scope,
                "Job_Type": job_type_scope,
                "Num_Instances": get_marker_arg_value(item, "multinode", num_instances),
                "Processor": get_marker_arg_value(item, "processor", processor_scope),
                "Integration": get_marker_arg_value(item, "integration", integration_scope),
                "Model": get_marker_arg_value(item, "model", model_scope),
                "GitHub_Link": github_link,
            }
        self.write_test_coverage_file(test_cov)

        if self.failure_conditions:
            message, total_issues = self.assemble_report_failure_message()
            if total_issues == 0:
                LOGGER.warning(f"Found failure message, but no issues. Message:\n{message}")
            else:
                raise TestReportGenerationFailure(message)


class TestReportGenerationFailure(Exception):
    pass


class RequiredMarkerNotFound(Exception):
    pass
