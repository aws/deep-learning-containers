import datetime
import os
import csv
import logging
import random
import re
import sys

import boto3
from botocore.config import Config
import docker
from fabric import Connection
import pytest

from test import test_utils
from test.test_utils import DEFAULT_REGION, UBUNTU_16_BASE_DLAMI, KEYS_TO_DESTROY_FILE
import test.test_utils.ec2 as ec2_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# Immutable constant for framework specific image fixtures
FRAMEWORK_FIXTURES = (
    "pytorch_inference",
    "pytorch_training",
    "mxnet_inference",
    "mxnet_training",
    "tensorflow_inference",
    "tensorflow_training",
    "training",
    "inference",
    "gpu",
    "cpu"
)

# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests", "*")]


def pytest_addoption(parser):
    default_images = test_utils.get_dlc_images()
    parser.addoption(
        "--images", default=default_images.split(" "), nargs="+", help="Specify image(s) to run",
    )
    parser.addoption(
        "--canary", action="store_true", default=False, help="Run canary tests",
    )
    parser.addoption(
        "--generate-coverage-doc", action="store_true", default=False, help="Generate a test coverage doc",
    )


@pytest.fixture(scope="function")
def num_nodes(request):
    return request.param


@pytest.fixture(scope="function")
def ec2_key_name(request):
    return request.param


@pytest.fixture(scope="session")
def region():
    return os.getenv("AWS_REGION", DEFAULT_REGION)


@pytest.fixture(scope="session")
def docker_client(region):
    test_utils.run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {region})", failure="Failed to log into ECR.",
    )
    return docker.from_env()


@pytest.fixture(scope="session")
def ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


@pytest.fixture(scope="session")
def ec2_resource(region):
    return boto3.resource("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


@pytest.fixture(scope="function")
def ec2_instance_type(request):
    return request.param if hasattr(request, "param") else "g4dn.xlarge"


@pytest.fixture(scope="function")
def ec2_instance_role_name(request):
    return request.param if hasattr(request, "param") else ec2_utils.EC2_INSTANCE_ROLE_NAME


@pytest.fixture(scope="function")
def ec2_instance_ami(request):
    return request.param if hasattr(request, "param") else UBUNTU_16_BASE_DLAMI


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ec2_instance(
        request, ec2_client, ec2_resource, ec2_instance_type, ec2_key_name, ec2_instance_role_name, ec2_instance_ami,
        region
):
    print(f"Creating instance: CI-CD {ec2_key_name}")
    key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
    params = {
        "KeyName": ec2_key_name,
        "ImageId": ec2_instance_ami,
        "InstanceType": ec2_instance_type,
        "IamInstanceProfile": {"Name": ec2_instance_role_name},
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"CI-CD {ec2_key_name}"}]},
        ],
        "MaxCount": 1,
        "MinCount": 1,
    }
    extra_volume_size_mapping = [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 300, }}]
    if ("benchmark" in os.getenv("TEST_TYPE") and (("mxnet_training" in request.fixturenames and "gpu_only" in request.fixturenames) or "mxnet_inference" in request.fixturenames)) \
            or ("tensorflow_training" in request.fixturenames and "gpu_only" in request.fixturenames and "horovod" in ec2_key_name):
        params["BlockDeviceMappings"] = extra_volume_size_mapping
    instances = ec2_resource.create_instances(**params)
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        if test_utils.is_pr_context():
            test_utils.destroy_ssh_keypair(ec2_client, key_filename)
        else:
            with open(KEYS_TO_DESTROY_FILE, "a") as destroy_keys:
                destroy_keys.write(f"{key_filename}\n")

    request.addfinalizer(terminate_ec2_instance)

    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(instance_id, system_status="ok", instance_status="ok", region=region)
    return instance_id, key_filename


@pytest.fixture(scope="function")
def ec2_connection(request, ec2_instance, ec2_key_name, region):
    """
    Fixture to establish connection with EC2 instance if necessary
    :param request: pytest test request
    :param ec2_instance: ec2_instance pytest fixture
    :param ec2_key_name: unique key name
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    instance_id, instance_pem_file = ec2_instance
    LOGGER.info(f"Instance ip_address: {ec2_utils.get_public_ip(instance_id, region)}")
    user = ec2_utils.get_instance_user(instance_id, region=region)
    conn = Connection(
        user=user,
        host=ec2_utils.get_public_ip(instance_id, region),
        connect_kwargs={"key_filename": [instance_pem_file]},
    )

    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 100000)

    artifact_folder = f"{ec2_key_name}-{unique_id}-folder"
    s3_test_artifact_location = test_utils.upload_tests_to_s3(artifact_folder)

    def delete_s3_artifact_copy():
        test_utils.delete_uploaded_tests_from_s3(s3_test_artifact_location)

    request.addfinalizer(delete_s3_artifact_copy)

    conn.run(f"aws s3 cp --recursive {test_utils.TEST_TRANSFER_S3_BUCKET}/{artifact_folder} $HOME/container_tests")
    conn.run(f"mkdir -p $HOME/container_tests/logs && chmod -R +x $HOME/container_tests/*")

    # Log into ECR if we are in canary context
    if test_utils.is_canary_context():
        public_registry = test_utils.PUBLIC_DLC_REGISTRY
        test_utils.login_to_ecr_registry(conn, public_registry, region)

    return conn


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


@pytest.fixture(scope="session")
def cpu_only():
    pass


@pytest.fixture(scope="session")
def gpu_only():
    pass


@pytest.fixture(scope="session")
def py3_only():
    pass


@pytest.fixture(scope="session")
def example_only():
    pass


def pytest_configure(config):
    # register canary marker
    config.addinivalue_line(
        "markers", "canary(message): mark test to run as a part of canary tests."
    )
    config.addinivalue_line(
        "markers", "integration(ml_integration): mark what the test is testing."
    )
    config.addinivalue_line(
        "markers", "model(model_name): name of the model being tested"
    )
    config.addinivalue_line(
        "markers", "multinode(num_instances): number of instances the test is run on, if not 1"
    )


def pytest_runtest_setup(item):
    if item.config.getoption("--canary"):
        canary_opts = [mark for mark in item.iter_markers(name="canary")]
        if not canary_opts:
            pytest.skip("Skipping non-canary tests")


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--generate-coverage-doc"):
        failure_conditions = {}
        test_coverage_file = test_utils.TEST_COVERAGE_FILE
        test_cov = {}
        for item in items:
            # Define additional csv options
            function_name = item.name.split("[")[0]
            function_key = f"{item.fspath}::{function_name}"
            str_fspath = str(item.fspath)
            str_keywords = str(item.keywords)

            # Construct Category and Github_Link fields based on the filepath
            category = str_fspath.split('/dlc_tests/')[-1].split('/')[0]
            github_link = f"https://github.com/aws/deep-learning-containers/blob/master/" \
                          f"{str_fspath.split('/deep-learning-containers/')[-1]}"

            # Only create a new test coverage item if we have not seen the function before. This is a necessary step,
            # as parametrization can make it appear as if the same test function is a unique test function
            if test_cov.get(function_key):
                continue

            # Based on keywords and filepaths, assign values
            scope = _infer_field_value("all", ("mxnet", "tensorflow", "pytorch"), str_fspath)
            train_inf = _infer_field_value("both", ("training", "inference"), str_fspath, str_keywords)
            integration = _infer_field_value("general integration",
                                             ("_dgl_", "smdebug", "gluonnlp", "smexperiments", "_mme_", "pipemode",
                                             "tensorboard", "_s3_"),
                                             str_keywords)
            model = _infer_field_value("N/A",
                                       ("mnist", "densenet", "squeezenet", "half_plus_two", "half_plus_three"),
                                       str_keywords)
            num_instances = _infer_field_value(1, ("_multinode_", "_multi-node_"), str_fspath, str_keywords)
            cpu_gpu = _infer_field_value("all", ("cpu", "gpu", "eia"), str_keywords)
            if cpu_gpu == "gpu":
                cpu_gpu = _handle_single_gpu_instances(function_key, str_keywords, failure_conditions)

            # Create a new test coverage item if we have not seen the function before. This is a necessary step,
            # as parametrization can make it appear as if the same test function is a unique test function
            test_cov[function_key] = {
                                        "Category": category,
                                        "Name": function_name,
                                        "Scope": scope,
                                        "Job_Type": train_inf,
                                        "Num_Instances": get_marker_arg_value(item, "multinode", num_instances),
                                        "Processor": cpu_gpu,
                                        "Integration": get_marker_arg_value(item, "integration", integration),
                                        "Model": get_marker_arg_value(item, "model", model),
                                        "Github_Link": github_link,
                                        }
        write_test_coverage_file(test_cov, test_coverage_file)

        if failure_conditions:
            message, total_issues = _assemble_report_failure_message(failure_conditions)
            if total_issues == 0:
                LOGGER.warning(f"Found failure message, but no issues. Message:\n{message}")
            else:
                raise TestReportGenerationFailure(message)


def _handle_single_gpu_instances(function_key, function_keywords, failures, cpu_gpu="gpu"):
    """
    Generally, we do not want tests running on single gpu instance types. However, there are exceptions to this rule.
    This function is used to determine whether we need to raise an error with report generation or not, based on
    whether we are using single gpu instanecs or not in a given test function.

    :param function_key: local/path/to/function::function_name
    :param function_keywords: string of keywords associated with the test function
    :param failures: running dictionary of failures associated with the github link
    :param cpu_gpu: whether the test is for cpu, gpu or both
    :return: cpu_gpu if not single gpu instance, else "single_gpu", and a dict with updated failure messages
    """

    # Define conditions where we allow a test function to run with a single gpu instance
    whitelist_single_gpu = False
    allowed_single_gpu = ("telemetry", "test_framework_version_gpu")

    # Regex in order to determine the gpu instance type
    gpu_instance_pattern = re.compile(r'\w+\.\d*xlarge')
    gpu_match = gpu_instance_pattern.search(function_keywords)

    if gpu_match:
        instance_type = gpu_match.group()
        num_gpus = ec2_utils.get_instance_num_gpus(instance_type=instance_type)

        for test in allowed_single_gpu:
            if test in function_key:
                whitelist_single_gpu = True
                break
        if num_gpus == 1:
            cpu_gpu = "single_gpu"
            if not whitelist_single_gpu:
                single_gpu_failure_message = f"Function {function_key} uses single-gpu instance type " \
                                             f"{instance_type}. Please use multi-gpu instance type."
                if not failures.get(function_key):
                    failures[function_key] = [single_gpu_failure_message]
                else:
                    failures[function_key].append(single_gpu_failure_message)

    return cpu_gpu, failures


def _assemble_report_failure_message(failure_messages):
    """
    Function to assemble the failure message if there are any to raise

    :param failure_messages: dict where key is the function, and value is a list of failures associated with the
    function
    :return: the final failure message string
    """
    final_message = ""
    total_issues = 0
    for func, messages in failure_messages.items():
        final_message += f"******Problems with {func}:******\n"
        for idx, message in enumerate(messages):
            final_message += f"{idx+1}. {message}\n"
            total_issues += 1
    final_message += f"TOTAL ISSUES: {total_issues}"

    return final_message, total_issues


def write_test_coverage_file(test_coverage_info, test_coverage_file):
    """
    Function to write out the test coverage file based on a dictionary defining key/value pairs of test coverage
    information

    :param test_coverage_info: dict representing the test coverage information
    :param test_coverage_file: outfile to write to
    """
    # Assemble the list of headers from one item in the dictionary
    field_names = []
    for _key, header in test_coverage_info.items():
        for field_name, _value in header.items():
            field_names.append(field_name)
        break

    # Write to the test coverage file
    with open(test_coverage_file, "w+") as tc_file:
        writer = csv.DictWriter(tc_file, delimiter=",", fieldnames=field_names)
        writer.writeheader()

        for _func_key, info in test_coverage_info.items():
            writer.writerow(info)


class TestReportGenerationFailure(Exception):
    pass


class RequiredMarkerNotFound(Exception):
    pass


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


def generate_unique_values_for_fixtures(metafunc_obj, images_to_parametrize, values_to_generate_for_fixture):
    """
    Take a dictionary (values_to_generate_for_fixture), that maps a fixture name used in a test function to another
    fixture that needs to be parametrized, and parametrize to create unique resources for a test.

    :param metafunc_obj: pytest metafunc object
    :param images_to_parametrize: <list> list of image URIs which are used in a test
    :param values_to_generate_for_fixture: <dict> Mapping of "Fixture used" -> "Fixture to be parametrized"
    :return: <dict> Mapping of "Fixture to be parametrized" -> "Unique values for fixture to be parametrized"
    """
    fixtures_parametrized = {}
    if images_to_parametrize:
        for key, new_fixture_name in values_to_generate_for_fixture.items():
            if key in metafunc_obj.fixturenames:
                fixtures_parametrized[new_fixture_name] = []
                for index, image in enumerate(images_to_parametrize):

                    # Tag fixtures with EC2 instance types if env variable is present
                    allowed_processors = ("gpu", "cpu", "eia")
                    instance_tag = ""
                    for processor in allowed_processors:
                        if processor in image:
                            instance_type = os.getenv(f"EC2_{processor.upper()}_INSTANCE_TYPE")
                            if instance_type:
                                instance_tag = f"-{instance_type.replace('.', '-')}"
                                break

                    image_tag = image.split(":")[-1].replace(".", "-")
                    fixtures_parametrized[new_fixture_name].append(
                        (
                            image,
                            f"{metafunc_obj.function.__name__}-{image_tag}-"
                            f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{index}{instance_tag}",
                        )
                    )
    return fixtures_parametrized


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                if lookup in image:
                    is_example_lookup = "example_only" in metafunc.fixturenames and "example" in image
                    is_standard_lookup = "example_only" not in metafunc.fixturenames and "example" not in image
                    if is_example_lookup or is_standard_lookup:
                        if "cpu_only" in metafunc.fixturenames and "cpu" in image:
                            images_to_parametrize.append(image)
                        elif "gpu_only" in metafunc.fixturenames and "gpu" in image:
                            images_to_parametrize.append(image)
                        elif "cpu_only" not in metafunc.fixturenames and "gpu_only" not in metafunc.fixturenames:
                            images_to_parametrize.append(image)

            # Remove all images tagged as "py2" if py3_only is a fixture
            if images_to_parametrize and "py3_only" in metafunc.fixturenames:
                images_to_parametrize = [py3_image for py3_image in images_to_parametrize if "py2" not in py3_image]

            # Parametrize tests that spin up an ecs cluster or tests that spin up an EC2 instance with a unique name
            values_to_generate_for_fixture = {
                "ecs_container_instance": "ecs_cluster_name",
                "ec2_connection": "ec2_key_name",
            }

            fixtures_parametrized = generate_unique_values_for_fixtures(
                metafunc, images_to_parametrize, values_to_generate_for_fixture
            )
            if fixtures_parametrized:
                for new_fixture_name, test_parametrization in fixtures_parametrized.items():
                    metafunc.parametrize(f"{fixture},{new_fixture_name}", test_parametrization)
            else:
                metafunc.parametrize(fixture, images_to_parametrize)

    # Parametrize for framework agnostic tests, i.e. sanity
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", images)
