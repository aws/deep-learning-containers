import os
import time

import boto3
import docker
import pytest

from test.test_utils import run_subprocess_cmd


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
def ecs_client():
    return boto3.client("ecs")


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
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    return instances[0]


@pytest.fixture(scope="session")
def ecs_cluster(request, ecs_client):
    """

    :param request:
    :param ec2_instance:
    :param ecs_client:
    :return:
    """
    cluster_name = f"{request.node.name}-ecs-cluster"
    ecs_client.create_cluster(
        clusterName=cluster_name
    )

    def delete_ecs_cluster():
        ecs_client.delete_cluster(cluster=cluster_name)

    request.addfinalizer(delete_ecs_cluster)

    # Wait for max 10 minutes for cluster status to be active
    timeout = time.time() + 600
    is_active = False
    while not is_active:
        if time.time() > timeout:
            raise TimeoutError(f"ECS cluster {cluster_name} timed out on creation")
        response = ecs_client.describe_clusters(clusters=[cluster_name])
        if response.get('clusters', [{}])[0].get('status') == 'ACTIVE':
            is_active = True

    return cluster_name


@pytest.mark.timeout(300)
@pytest.fixture(scope="session")
def ecs_container_instance(request, ecs_cluster, ec2_client, ec2_resource, ec2_instance_type):
    user_data = f"{ecs_cluster}.txt"
    with open(user_data, 'w') as user_data_file:
        user_data_file.write(f"#!/bin/bash\necho ECS_CLUSTER={ecs_cluster} >> /etc/ecs/ecs.config")

    instances = ec2_resource.create_instances(
        KeyName="pytest.pem",
        ImageId=UBUNTU_16_BASE_DLAMI,
        # hard coding for now
        InstanceType="p2.8xlarge",
        MaxCount=1,
        MinCount=1,
        UserData=f"file://{user_data}",
        IamInstanceProfile={"Name": "ecsInstanceRole"}
    )
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    return instances[0], ecs_cluster


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
