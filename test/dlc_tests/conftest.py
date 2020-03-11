import os

import boto3
import docker
import pytest

from test.dlc_tests.test_utils.general import run_subprocess_cmd


# Constant to represent AMI Id used to spin up EC2 instances
UBUNTU_16_BASE_DLAMI = "ami-0e57002aaafd42113"


# Immutable constant for framework specific image fixtures
FRAMEWORK_FIXTURES = (
    "pytorch_inference",
    "pytorch_training",
    "mxnet_inference",
    "mxnet_training",
    "tensorflow_inference",
    "tensorflow_training",
)


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests", "*")]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        default=os.getenv("DLC_IMAGES").split(" "),
        nargs="+",
        help="Specify image(s) to run",
    )
    parser.addoption(
        "--ec2-instance-type", required=False, help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {os.getenv('AWS_REGION', 'us-west-2')})",
        failure="Failed to log into ECR.",
    )
    return docker.from_env()


@pytest.fixture(scope="session")
def ec2_client():
    return boto3.client("ec2")


@pytest.fixture(scope="session")
def ec2_resource():
    return boto3.resource("ec2")


@pytest.fixture(scope="session")
def ec2_instance_type(request):
    return request.config.getoption("--ec2-instance-type")


@pytest.mark.timeout(300)
@pytest.fixture(scope="session")
def ec2_instance(request, ec2_client, ec2_instance_type, ec2_resource):
    instances = ec2_resource.create_instances(
        KeyName="pytest.pem",
        ImageId=UBUNTU_16_BASE_DLAMI,
        InstanceType=ec2_instance_type,
        MaxCount=1,
        MinCount=1,
    )
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    return instances[0]


@pytest.fixture(scope="session")
def ec2_connection(ec2_instance):
    """
    Not implemented fixture to establish connection with EC2 instance if necessary

    :param ec2_instance: ec2_instance pytest fixture
    :return: Fabric connection object
    """
    # With Connection(ec2_instance.public_ip_address_ as c:
    #     yield c
    pass


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                if lookup in image:
                    images_to_parametrize.append(image)
            metafunc.parametrize(fixture, images_to_parametrize)

    # Parametrize for framework agnostic tests, i.e. sanity
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", images)
