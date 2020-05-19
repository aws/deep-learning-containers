import datetime
import os
import logging
import random
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
    "inference"
)


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests", "*")]


def pytest_addoption(parser):
    default_images = test_utils.get_dlc_images()
    parser.addoption(
        "--images", default=default_images.split(" "), nargs="+", help="Specify image(s) to run",
    )


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
    return boto3.client("ec2", region_name=region, config=Config(retries={'max_attempts': 10}))


@pytest.fixture(scope="session")
def ec2_resource(region):
    return boto3.resource("ec2", region_name=region, config=Config(retries={'max_attempts': 10}))


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
        request, ec2_client, ec2_resource, ec2_instance_type, ec2_key_name, ec2_instance_role_name, ec2_instance_ami, region
):
    print(f"Creating instance: CI-CD {ec2_key_name}")
    key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
    extra_volume_size_mapping = [
                        {
                            'DeviceName': '/dev/sda1',
                            'Ebs': {
                                'VolumeSize': 300,
                            },
                        },
                    ]
    if ("performance" in ec2_key_name) and ("mxnet" in ec2_key_name) and ("training_cpu" not in ec2_key_name):
        instances = ec2_resource.create_instances(
            BlockDeviceMappings=extra_volume_size_mapping,
            KeyName=ec2_key_name,
            ImageId=ec2_instance_ami,
            InstanceType=ec2_instance_type,
            IamInstanceProfile={"Name": ec2_instance_role_name},
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": f"CI-CD {ec2_key_name}"}]
                },
            ],
            MaxCount=1,
            MinCount=1,
        )
    else:
        instances = ec2_resource.create_instances(
            KeyName=ec2_key_name,
            ImageId=ec2_instance_ami,
            InstanceType=ec2_instance_type,
            IamInstanceProfile={"Name": ec2_instance_role_name},
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": f"CI-CD {ec2_key_name}"}]
                },
            ],
            MaxCount=1,
            MinCount=1,
        )
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        if test_utils.is_pr_context():
            test_utils.destroy_ssh_keypair(ec2_client, key_filename)
        else:
            with open(KEYS_TO_DESTROY_FILE, 'a') as destroy_keys:
                destroy_keys.write(f"{key_filename}\n")

    request.addfinalizer(terminate_ec2_instance)

    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(
        instance_id, system_status="ok", instance_status="ok", region=region
    )
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
        connect_kwargs={"key_filename": [instance_pem_file]}
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
                    image_tag = image.split(":")[-1].replace(".", "-")
                    fixtures_parametrized[new_fixture_name].append(
                        (
                            image,
                            f"{metafunc_obj.function.__name__}-{image_tag}-"
                            f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{index}",
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
                images_to_parametrize = [py3_image for py3_image in images_to_parametrize if 'py2' not in py3_image]

            # Parametrize tests that spin up an ecs cluster or tests that spin up an EC2 instance with a unique name
            values_to_generate_for_fixture = {"ecs_container_instance": "ecs_cluster_name",
                                              "ec2_connection": "ec2_key_name"}

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
