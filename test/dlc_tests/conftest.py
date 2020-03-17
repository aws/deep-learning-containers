import os

import boto3
import docker
import pytest

from test.test_utils import run_subprocess_cmd, UBUNTU_16_BASE_DLAMI, DEFAULT_REGION


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


@pytest.fixture(scope="session")
def region():
    return os.getenv('AWS_REGION', DEFAULT_REGION)


@pytest.fixture(scope="session")
def docker_client(region):
    run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {region})",
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
    return request.param


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
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

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


@pytest.fixture(scope="session")
def cpu_only():
    pass


@pytest.fixture(scope="session")
def gpu_only():
    pass


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                image_tag = image.split(':')[-1]
                if lookup in image:
                    if "cpu_only" in metafunc.fixturenames and "cpu" in image:
                        images_to_parametrize.append(image)
                    elif "gpu_only" in metafunc.fixturenames and "gpu" in image:
                        images_to_parametrize.append(image)
                    elif (
                        "cpu_only" not in metafunc.fixturenames
                        and "gpu_only" not in metafunc.fixturenames
                    ):
                        images_to_parametrize.append(image)

            if os.getenv("TEST_TYPE") == "ECS" and images_to_parametrize:
                ecs_parametrization = []
                for image in images_to_parametrize:
                    image_tag = image.split(':')
                    ecs_parametrization.append((image, f"{metafunc.function.__name__}-{image_tag}"))
                metafunc.parametrize(f"{fixture},ecs_cluster_name", ecs_parametrization)
            else:
                metafunc.parametrize(fixture, images_to_parametrize)

    # Parametrize for framework agnostic tests, i.e. sanity
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", images)
