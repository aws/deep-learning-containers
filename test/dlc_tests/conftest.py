import os

import boto3
from botocore.config import Config
import docker
from fabric import Connection
import pytest

from test.test_utils import destroy_ssh_keypair, generate_ssh_keypair, run_subprocess_cmd
from test.test_utils import DEFAULT_REGION, UBUNTU_16_BASE_DLAMI
import test.test_utils.ec2 as ec2_utils


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
        "--images", default=os.getenv("DLC_IMAGES").split(" "), nargs="+", help="Specify image(s) to run",
    )


@pytest.fixture(scope="function")
def ec2_key_name(request):
    return request.param


@pytest.fixture(scope="session")
def region():
    return os.getenv("AWS_REGION", DEFAULT_REGION)


@pytest.fixture(scope="session")
def docker_client(region):
    run_subprocess_cmd(
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
    return request.param if hasattr(request, "param") else ""


@pytest.fixture(scope="function")
def ec2_instance_ami(request):
    return request.param if hasattr(request, "param") else UBUNTU_16_BASE_DLAMI


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ec2_instance(
        request, ec2_client, ec2_resource, ec2_instance_type, ec2_key_name, ec2_instance_role_name, ec2_instance_ami, region
):
    print(f"Creating instance: CI-CD {ec2_key_name}")
    key_filename = generate_ssh_keypair(ec2_client, ec2_key_name)
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
        destroy_ssh_keypair(ec2_client, key_filename)

    request.addfinalizer(terminate_ec2_instance)

    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(
        instance_id, system_status="ok", instance_status="ok", region=region
    )
    return instance_id, key_filename


@pytest.fixture(scope="function")
def ec2_connection(ec2_instance, region):
    """
    Fixture to establish connection with EC2 instance if necessary
    :param ec2_instance: ec2_instance pytest fixture
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    instance_id, instance_pem_file = ec2_instance
    user = ec2_utils.get_instance_user(instance_id, region=region)
    conn = Connection(
        user=user,
        host=ec2_utils.get_public_ip(instance_id, region),
        connect_kwargs={"key_filename": [instance_pem_file]}
    )
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


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                if lookup in image:
                    if "cpu_only" in metafunc.fixturenames and "cpu" in image:
                        images_to_parametrize.append(image)
                    elif "gpu_only" in metafunc.fixturenames and "gpu" in image:
                        images_to_parametrize.append(image)
                    elif "cpu_only" not in metafunc.fixturenames and "gpu_only" not in metafunc.fixturenames:
                        images_to_parametrize.append(image)

            # Remove all images tagged as "py2" if py3_only is a fixture
            if images_to_parametrize and "py3_only" in metafunc.fixturenames:
                images_to_parametrize = [py3_image for py3_image in images_to_parametrize if 'py2' not in py3_image]

            # Parametrize tests that spin up an ecs cluster with unique name
            if (images_to_parametrize and
                    ("ecs_container_instance" in metafunc.fixturenames or "ec2_connection" in metafunc.fixturenames)):
                test_parametrization = []
                for index, image in enumerate(images_to_parametrize):
                    image_tag = image.split(":")[-1].replace(".", "-")
                    test_parametrization.append(
                        (
                            image,
                            f"{metafunc.function.__name__}-{image_tag}-"
                            f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{index}",
                        )
                    )
                new_fixture_name = ("ecs_cluster_name" if "ecs_container_instance" in metafunc.fixturenames else
                                    "ec2_key_name")
                metafunc.parametrize(f"{fixture},{new_fixture_name}", test_parametrization)
            else:
                metafunc.parametrize(fixture, images_to_parametrize)

    # Parametrize for framework agnostic tests, i.e. sanity
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", images)
