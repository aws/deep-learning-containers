import datetime
import os
import logging
import random
import sys
import re

import boto3
from botocore.exceptions import ClientError
import docker
import pytest

from botocore.config import Config
from fabric import Connection

import test.test_utils.ec2 as ec2_utils

from test import test_utils
from test.test_utils import (
    is_benchmark_dev_context,
    get_framework_and_version_from_tag,
    get_job_type_from_image,
    is_tf_version,
    is_below_framework_version,
    is_e3_image,
    is_sagemaker_image,
    DEFAULT_REGION,
    P3DN_REGION,
    UBUNTU_18_BASE_DLAMI_US_EAST_1,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_EAST_1,
    AML2_GPU_DLAMI_US_WEST_2,
    AML2_GPU_DLAMI_US_EAST_1,
    KEYS_TO_DESTROY_FILE,
    are_efa_tests_disabled,
    get_ecr_repo_name,
    UBUNTU_HOME_DIR,
)
from test.test_utils.test_reporting import TestReportGenerator

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# Immutable constant for framework specific image fixtures
FRAMEWORK_FIXTURES = (
    # ECR repo name fixtures
    # PyTorch
    "pytorch_training",
    "pytorch_training_habana",
    "pytorch_inference",
    "pytorch_inference_eia",
    "pytorch_inference_neuron",
    "pytorch_inference_graviton",
    # TensorFlow
    "tensorflow_training",
    "tensorflow_inference",
    "tensorflow_inference_eia",
    "tensorflow_inference_neuron",
    "tensorflow_training_habana",
    "tensorflow_inference_graviton",
    # MxNET
    "mxnet_training",
    "mxnet_inference",
    "mxnet_inference_eia",
    "mxnet_inference_neuron",
    "mxnet_inference_graviton",
    # HuggingFace
    "huggingface_tensorflow_training",
    "huggingface_pytorch_training",
    "huggingface_mxnet_training",
    "huggingface_tensorflow_inference",
    "huggingface_pytorch_inference",
    "huggingface_mxnet_inference",
    "huggingface_tensorflow_trcomp_training",
    "huggingface_pytorch_trcomp_training",
    # Autogluon
    "autogluon_training",
    # Processor fixtures
    "gpu",
    "cpu",
    "eia",
    "neuron",
    "hpu",
    # Architecture
    "graviton",
    # Job Type fixtures
    "training",
    "inference",
)

# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests")]


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
    parser.addoption(
        "--multinode", action="store_true", default=False, help="Run only multi-node tests",
    )
    parser.addoption(
        "--efa", action="store_true", default=False, help="Run only efa tests",
    )
    parser.addoption(
        "--quick_checks", action="store_true", default=False, help="Run quick check tests",
    )


@pytest.fixture(scope="function")
def num_nodes(request):
    return request.param


@pytest.fixture(scope="function")
def ec2_key_name(request):
    return request.param

@pytest.fixture(scope="function")
def ec2_key_file_name(request):
    return request.param

@pytest.fixture(scope="function")
def ec2_user_name(request):
    return request.param

@pytest.fixture(scope="function")
def ec2_public_ip(request):
    return request.param

@pytest.fixture(scope="function")
def ec2_key_file_name(request):
    return request.param


@pytest.fixture(scope="function")
def ec2_user_name(request):
    return request.param


@pytest.fixture(scope="function")
def ec2_public_ip(request):
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
def ecr_client(region):
    return boto3.client("ecr", region_name=region)


@pytest.fixture(scope="session")
def sts_client(region):
    return boto3.client("sts", region_name=region)


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
    return request.param if hasattr(request, "param") else UBUNTU_18_BASE_DLAMI_US_WEST_2


@pytest.fixture(scope="function")
def ei_accelerator_type(request):
    return request.param if hasattr(request, "param") else None


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ec2_instance(
    request,
    ec2_client,
    ec2_resource,
    ec2_instance_type,
    ec2_key_name,
    ec2_instance_role_name,
    ec2_instance_ami,
    region,
    ei_accelerator_type,
):
    if ec2_instance_type == "p3dn.24xlarge":
        region = P3DN_REGION
        ec2_client = boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))
        ec2_resource = boto3.resource("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))
        if ec2_instance_ami != PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_EAST_1:
            ec2_instance_ami = (
                AML2_GPU_DLAMI_US_EAST_1
                if ec2_instance_ami == AML2_GPU_DLAMI_US_WEST_2
                else UBUNTU_18_BASE_DLAMI_US_EAST_1
            )

    print(f"Creating instance: CI-CD {ec2_key_name}")
    key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)

    def delete_ssh_keypair():
        if test_utils.is_pr_context():
            test_utils.destroy_ssh_keypair(ec2_client, key_filename)
        else:
            with open(KEYS_TO_DESTROY_FILE, "a") as destroy_keys:
                destroy_keys.write(f"{key_filename}\n")

    request.addfinalizer(delete_ssh_keypair)

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

    volume_name = "/dev/sda1" if ec2_instance_ami in test_utils.UL_AMI_LIST else "/dev/xvda"

    if (
        "pytorch_training_habana" in request.fixturenames
        or "tensorflow_training_habana" in request.fixturenames
        or "hpu" in request.fixturenames
    ):
        user_data = """#!/bin/bash
        sudo apt-get update && sudo apt-get install -y awscli"""
        params["UserData"] = user_data
        params["BlockDeviceMappings"] = [{"DeviceName": volume_name, "Ebs": {"VolumeSize": 1000,},}]
    elif (
        (
            ("benchmark" in os.getenv("TEST_TYPE") or is_benchmark_dev_context())
            and (
                ("mxnet_training" in request.fixturenames and "gpu_only" in request.fixturenames)
                or "mxnet_inference" in request.fixturenames
            )
        )
        or (is_neuron_image)
        or (
            "tensorflow_training" in request.fixturenames
            and "gpu_only" in request.fixturenames
            and "horovod" in ec2_key_name
        )
        or ("tensorflow_inference" in request.fixturenames and "graviton_compatible_only" in request.fixturenames)
        or ("graviton" in request.fixturenames)
    ):
        params["BlockDeviceMappings"] = [{"DeviceName": volume_name, "Ebs": {"VolumeSize": 300,},}]
    else:
        # Using private AMI, the EBS volume size is reduced to 28GB as opposed to 50GB from public AMI. This leads to space issues on test instances
        # TODO: Revert the configuration once DLAMI is public
        params["BlockDeviceMappings"] = [{"DeviceName": volume_name, "Ebs": {"VolumeSize": 90,},}]
    if ei_accelerator_type:
        params["ElasticInferenceAccelerators"] = [{"Type": ei_accelerator_type, "Count": 1}]
        availability_zones = {
            "us-west-2": ["us-west-2a", "us-west-2b", "us-west-2c"],
            "us-east-1": ["us-east-1a", "us-east-1b", "us-east-1c"],
        }
        for a_zone in availability_zones[region]:
            params["Placement"] = {"AvailabilityZone": a_zone}
            try:
                instances = ec2_resource.create_instances(**params)
                if instances:
                    break
            except ClientError as e:
                LOGGER.error(f"Failed to launch in {a_zone} due to {e}")
                continue
    else:
        try:
            instances = ec2_resource.create_instances(**params)
        except ClientError as e:
            if e.response["Error"]["Code"] == "InsufficientInstanceCapacity":
                LOGGER.warning(f"Failed to launch {ec2_instance_type} in {region} because of insufficient capacity")
                if ec2_instance_type in ec2_utils.ICE_SKIP_INSTANCE_LIST:
                    pytest.skip(f"Skipping test because {ec2_instance_type} instance could not be launched.")
            raise
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(instance_id, system_status="ok", instance_status="ok", region=region)
    return instance_id, key_filename


def is_neuron_image(fixtures):
    """
    Returns true if a neuron fixture is present in request.fixturenames
    :param request.fixturenames: active fixtures in the request
    :return: bool
    """
    neuron_fixtures = ["tensorflow_inference_neuron", "mxnet_inference_neuron", "pytorch_inference_neuron"]

    for fixture in neuron_fixtures:
        if fixture in fixtures:
            return True
    return False


@pytest.fixture(scope="function")
def ec2_connection(request, ec2_instance, ec2_key_name, ec2_instance_type, region):
    """
    Fixture to establish connection with EC2 instance if necessary
    :param request: pytest test request
    :param ec2_instance: ec2_instance pytest fixture
    :param ec2_key_name: unique key name
    :param ec2_instance_type: ec2_instance_type pytest fixture
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    instance_id, instance_pem_file = ec2_instance
    region = P3DN_REGION if ec2_instance_type == "p3dn.24xlarge" else region
    ip_address = ec2_utils.get_public_ip(instance_id, region=region)
    LOGGER.info(f"Instance ip_address: {ip_address}")
    user = ec2_utils.get_instance_user(instance_id, region=region)
    LOGGER.info(f"Connecting to {user}@{ip_address}")
    conn = Connection(
        user=user, host=ip_address, connect_kwargs={"key_filename": [instance_pem_file]}, connect_timeout=18000,
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


@pytest.fixture(scope="function")
def upload_habana_test_artifact(request, ec2_connection):
    """
    Fixture to upload the habana test repo to ec2 instance
    :param request: pytest test request
    :param ec2_connection: fabric connection object
    :return: None
    """
    habana_test_repo = "gaudi-test-suite.tar.gz"
    ec2_connection.put(habana_test_repo, f"{UBUNTU_HOME_DIR}")
    ec2_connection.run(f"tar -xvf {habana_test_repo}")


@pytest.fixture(scope="function")
def existing_ec2_instance_connection(request, ec2_key_file_name, ec2_user_name, ec2_public_ip):
    """
    Fixture to establish connection with EC2 instance if necessary
    :param request: pytest test request
    :param ec2_key_file_name: ec2 key file name
    :param ec2_user_name: username of the ec2 instance to login
    :param ec2_public_ip: public ip address of the instance
    :return: Fabric connection object
    """
    conn = Connection(user=ec2_user_name, host=ec2_public_ip, connect_kwargs={"key_filename": [ec2_key_file_name]},)

    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 100000)
    ec2_key_name = ec2_public_ip.split(".")[0]
    artifact_folder = f"{ec2_key_name}-{unique_id}-folder"
    s3_test_artifact_location = test_utils.upload_tests_to_s3(artifact_folder)

    def delete_s3_artifact_copy():
        test_utils.delete_uploaded_tests_from_s3(s3_test_artifact_location)

    request.addfinalizer(delete_s3_artifact_copy)

    conn.run(f"aws s3 cp --recursive {test_utils.TEST_TRANSFER_S3_BUCKET}/{artifact_folder} $HOME/container_tests")
    conn.run(f"mkdir -p $HOME/container_tests/logs && chmod -R +x $HOME/container_tests/*")

    return conn


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


@pytest.fixture(scope="session")
def non_huggingface_only():
    pass


@pytest.fixture(scope="session")
def training_compiler_only():
    pass


@pytest.fixture(scope="session")
def non_autogluon_only():
    pass


@pytest.fixture(scope="session")
def cpu_only():
    pass


@pytest.fixture(scope="session")
def gpu_only():
    pass


@pytest.fixture(scope="session")
def x86_compatible_only():
    pass


@pytest.fixture(scope="session")
def graviton_compatible_only():
    pass


@pytest.fixture(scope="session")
def sagemaker():
    pass


@pytest.fixture(scope="session")
def sagemaker_only():
    pass


@pytest.fixture(scope="session")
def py3_only():
    pass


@pytest.fixture(scope="session")
def example_only():
    pass


@pytest.fixture(scope="session")
def huggingface_only():
    pass


@pytest.fixture(scope="session")
def huggingface():
    pass


@pytest.fixture(scope="session")
def tf2_only():
    pass


@pytest.fixture(scope="session")
def tf23_and_above_only():
    pass


@pytest.fixture(scope="session")
def tf24_and_above_only():
    pass


@pytest.fixture(scope="session")
def tf25_and_above_only():
    pass


@pytest.fixture(scope="session")
def tf21_and_above_only():
    pass


@pytest.fixture(scope="session")
def mx18_and_above_only():
    pass


@pytest.fixture(scope="session")
def pt111_and_above_only():
    pass


@pytest.fixture(scope="session")
def pt17_and_above_only():
    pass


@pytest.fixture(scope="session")
def pt16_and_above_only():
    pass


@pytest.fixture(scope="session")
def pt15_and_above_only():
    pass


@pytest.fixture(scope="session")
def pt14_and_above_only():
    pass


def framework_version_within_limit(metafunc_obj, image):
    """
    Test all pytest fixtures for TensorFlow version limits, and return True if all requirements are satisfied

    :param metafunc_obj: pytest metafunc object from which fixture names used by test function will be obtained
    :param image: Image URI for which the validation must be performed
    :return: True if all validation succeeds, else False
    """
    image_framework_name, _ = get_framework_and_version_from_tag(image)
    if image_framework_name == "tensorflow":
        tf2_requirement_failed = "tf2_only" in metafunc_obj.fixturenames and not is_tf_version("2", image)
        tf25_requirement_failed = "tf25_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "2.5", image, "tensorflow"
        )
        tf24_requirement_failed = "tf24_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "2.4", image, "tensorflow"
        )
        tf23_requirement_failed = "tf23_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "2.3", image, "tensorflow"
        )
        tf21_requirement_failed = "tf21_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "2.1", image, "tensorflow"
        )
        if (
            tf2_requirement_failed
            or tf21_requirement_failed
            or tf24_requirement_failed
            or tf25_requirement_failed
            or tf23_requirement_failed
        ):
            return False
    if image_framework_name == "mxnet":
        mx18_requirement_failed = "mx18_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.8", image, "mxnet"
        )
        if mx18_requirement_failed:
            return False
    if image_framework_name == "pytorch":
        pt111_requirement_failed = "pt111_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.11", image, "pytorch"
        )
        pt17_requirement_failed = "pt17_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.7", image, "pytorch"
        )
        pt16_requirement_failed = "pt16_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.6", image, "pytorch"
        )
        pt15_requirement_failed = "pt15_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.5", image, "pytorch"
        )
        pt14_requirement_failed = "pt14_and_above_only" in metafunc_obj.fixturenames and is_below_framework_version(
            "1.4", image, "pytorch"
        )
        if pt111_requirement_failed or pt17_requirement_failed or pt16_requirement_failed or pt15_requirement_failed or pt14_requirement_failed:
            return False
    return True


def pytest_configure(config):
    # register canary marker
    config.addinivalue_line("markers", "canary(message): mark test to run as a part of canary tests.")
    config.addinivalue_line("markers", "quick_check(message): mark test to run as a part of quick check tests.")
    config.addinivalue_line("markers", "integration(ml_integration): mark what the test is testing.")
    config.addinivalue_line("markers", "model(model_name): name of the model being tested")
    config.addinivalue_line("markers", "multinode(num_instances): number of instances the test is run on, if not 1")
    config.addinivalue_line("markers", "processor(cpu/gpu/eia/hpu): explicitly mark which processor is used")
    config.addinivalue_line("markers", "efa(): explicitly mark to run efa tests")


def pytest_runtest_setup(item):
    """
    Handle custom markers and options
    """
    # Handle quick check tests
    quick_checks_opts = [mark for mark in item.iter_markers(name="quick_checks")]
    # On PR, skip quick check tests unless we are on quick_checks job
    test_type = os.getenv("TEST_TYPE", "UNDEFINED")
    quick_checks_test_type = "quick_checks"
    if test_type != quick_checks_test_type and test_utils.is_pr_context():
        if quick_checks_opts:
            pytest.skip(
                f"Skipping quick check tests on PR, since test type is {test_type}, and not {quick_checks_test_type}"
            )

    # If we have enabled the quick_checks flag, we expect to only run tests marked as quick_check
    if item.config.getoption("--quick_checks"):
        if not quick_checks_opts:
            pytest.skip("Skipping non-quick-check tests")

    # Handle canary test conditional skipping
    if item.config.getoption("--canary"):
        canary_opts = [mark for mark in item.iter_markers(name="canary")]
        if not canary_opts:
            pytest.skip("Skipping non-canary tests")

    # Handle multinode conditional skipping
    if item.config.getoption("--multinode"):
        multinode_opts = [mark for mark in item.iter_markers(name="multinode")]
        if not multinode_opts:
            pytest.skip("Skipping non-multinode tests")

    # Handle efa conditional skipping
    if item.config.getoption("--efa"):
        efa_tests = [mark for mark in item.iter_markers(name="efa")]
        if not efa_tests:
            pytest.skip("Skipping non-efa tests")


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--generate-coverage-doc"):
        report_generator = TestReportGenerator(items)
        report_generator.generate_coverage_doc()
        report_generator.generate_sagemaker_reports()


def generate_unique_values_for_fixtures(metafunc_obj, images_to_parametrize, values_to_generate_for_fixture):
    """
    Take a dictionary (values_to_generate_for_fixture), that maps a fixture name used in a test function to another
    fixture that needs to be parametrized, and parametrize to create unique resources for a test.

    :param metafunc_obj: pytest metafunc object
    :param images_to_parametrize: <list> list of image URIs which are used in a test
    :param values_to_generate_for_fixture: <dict> Mapping of "Fixture used" -> "Fixture to be parametrized"
    :return: <dict> Mapping of "Fixture to be parametrized" -> "Unique values for fixture to be parametrized"
    """
    job_type_map = {"training": "tr", "inference": "inf"}
    framework_name_map = {
        "tensorflow": "tf",
        "mxnet": "mx",
        "pytorch": "pt",
        "huggingface_pytorch": "hf-pt",
        "huggingface_tensorflow": "hf-tf",
        "huggingface_pytorch_trcomp": "hf-pt-trc",
        "huggingface_tensorflow_trcomp": "hf-tf-trc",
        "autogluon": "ag",
    }
    fixtures_parametrized = {}

    if images_to_parametrize:
        for key, new_fixture_name in values_to_generate_for_fixture.items():
            if key in metafunc_obj.fixturenames:
                fixtures_parametrized[new_fixture_name] = []
                for index, image in enumerate(images_to_parametrize):

                    # Tag fixtures with EC2 instance types if env variable is present
                    allowed_processors = ("gpu", "cpu", "eia", "neuron", "hpu")
                    instance_tag = ""
                    for processor in allowed_processors:
                        if processor in image:
                            if "graviton" in image:
                                instance_type_env = f"EC2_{processor.upper()}_GRAVITON_INSTANCE_TYPE"
                            else:
                                instance_type_env = f"EC2_{processor.upper()}_INSTANCE_TYPE"
                            instance_type = os.getenv(instance_type_env)
                            if instance_type:
                                instance_tag = f"-{instance_type.replace('.', '-')}"
                                break

                    image_tag = image.split(":")[-1].replace(".", "-")

                    framework, _ = get_framework_and_version_from_tag(image)

                    job_type = get_job_type_from_image(image)

                    fixtures_parametrized[new_fixture_name].append(
                        (
                            image,
                            f"{metafunc_obj.function.__name__}-{framework_name_map.get(framework)}-"
                            f"{job_type_map.get(job_type)}-{image_tag}-"
                            f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{index}{instance_tag}",
                        )
                    )
    return fixtures_parametrized


def lookup_condition(lookup, image):
    """
    Return true if the ECR repo name ends with the lookup or lookup contains job type or device type part of the image uri.
    """
    #Extract ecr repo name from the image and check if it exactly matches the lookup (fixture name)
    repo_name = image.split("/")[-1].split(":")[0]

    job_type = ("training", "inference",)
    device_type = ("cpu", "gpu", "eia", "neuron", "hpu")

    if not repo_name.endswith(lookup):
        if (lookup in job_type or lookup in device_type) and lookup in image:
            return True
        # Pytest does not allow usage of fixtures, specially dynamically loaded fixtures into pytest.mark.parametrize
        # See https://github.com/pytest-dev/pytest/issues/349.
        # Hence, explicitly setting the below fixtues to allow trcomp images to run on E3 test
        elif "huggingface-pytorch-trcomp-training" in repo_name:
            if lookup == "pytorch-training":
                return True
        elif "huggingface-tensorflow-trcomp-training" in repo_name:
            if lookup == "tensorflow-training":
                return True
        else:
            return False
    else:
        return True


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Don't parametrize if there are no images to parametrize
    if not images:
        return

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                if lookup_condition(lookup, image):
                    is_example_lookup = "example_only" in metafunc.fixturenames and "example" in image
                    is_huggingface_lookup = (
                            ("huggingface_only" in metafunc.fixturenames or "huggingface" in metafunc.fixturenames)
                            and "huggingface" in image
                    )
                    is_trcomp_lookup = "trcomp" in image and all(
                        fixture_name not in metafunc.fixturenames
                        for fixture_name in ["example_only"]
                    )
                    is_standard_lookup = all(
                        fixture_name not in metafunc.fixturenames
                        for fixture_name in ["example_only", "huggingface_only"]
                    ) and all(keyword not in image for keyword in ["example", "huggingface"])
                    if "sagemaker_only" in metafunc.fixturenames and is_e3_image(image):
                        LOGGER.info(f"Not running E3 image {image} on sagemaker_only test")
                        continue
                    if is_sagemaker_image(image):
                        if "sagemaker_only" not in metafunc.fixturenames and "sagemaker" not in metafunc.fixturenames:
                            LOGGER.info(
                                f"Skipping test, as this function is not marked as 'sagemaker_only' or 'sagemaker'"
                            )
                            continue
                    if not framework_version_within_limit(metafunc, image):
                        continue
                    if "non_huggingface_only" in metafunc.fixturenames and "huggingface" in image:
                        continue
                    if "non_autogluon_only" in metafunc.fixturenames and "autogluon" in image:
                        continue
                    if "x86_compatible_only" in metafunc.fixturenames and "graviton" in image:
                        continue
                    if "training_compiler_only" in metafunc.fixturenames and not ("trcomp" in image):
                        continue
                    if is_example_lookup or is_huggingface_lookup or is_standard_lookup or is_trcomp_lookup:
                        if "cpu_only" in metafunc.fixturenames and "cpu" in image and "eia" not in image:
                            images_to_parametrize.append(image)
                        elif "gpu_only" in metafunc.fixturenames and "gpu" in image:
                            images_to_parametrize.append(image)
                        elif "graviton_compatible_only" in metafunc.fixturenames and "graviton" in image:
                            images_to_parametrize.append(image)
                        elif (
                            "cpu_only" not in metafunc.fixturenames
                            and "gpu_only" not in metafunc.fixturenames
                            and "graviton_compatible_only" not in metafunc.fixturenames
                        ):
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


@pytest.fixture(autouse=True)
def skip_efa_tests(request):
    efa_tests = [mark for mark in request.node.iter_markers(name="efa")]

    if efa_tests and are_efa_tests_disabled():
        pytest.skip("Skipping EFA tests as EFA tests are disabled.")


@pytest.fixture(autouse=True)
def disable_test(request):
    test_name = request.node.name
    # We do not have a regex pattern to find CB name, which means we must resort to string splitting
    build_arn = os.getenv("CODEBUILD_BUILD_ARN")
    build_name = build_arn.split("/")[-1].split(":")[0] if build_arn else None
    version = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")

    if test_utils.is_test_disabled(test_name, build_name, version):
        pytest.skip(f"Skipping {test_name} test because it has been disabled.")
