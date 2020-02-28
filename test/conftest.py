import os

import boto3
import docker
import pytest


# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join('container_tests', '*')]


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        required=True,
        nargs='+',
        help="Specify image(s) to run"
    )
    parser.addoption(
        "--ec2-instance-type",
        required=False,
        help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


@pytest.fixture(scope="session")
def ec2_client():
    return boto3.client('ec2')


@pytest.fixture(scope="session")
def ec2_resource():
    return boto3.resource('ec2')


@pytest.fixture(scope="session")
def ec2_instance_type(request):
    return request.config.getoption("--ec2-instance-type")


@pytest.mark.timeout(300)
@pytest.fixture(scope="session")
def start_ec2_instance(request, ec2_client, ec2_instance_type, ec2_resource):
    instances = ec2_resource.create_instances(
        KeyName="pytest.pem",
        ImageId='ami-0e57002aaafd42113',
        InstanceType=ec2_instance_type,
        MaxCount=1,
        MinCount=1
    )
    instance_id = instances[0].id

    # Define finalizer to terminate instance after this fixture completes
    def terminate():
        ec2_client.terminate_instances(InstanceIds=[instance_id])
    request.addfinalizer(terminate)

    waiter = ec2_client.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    return instances[0]


@pytest.fixture(scope="session")
def run_on_ec2_instance(request, start_ec2_instance):
    pass


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


def pytest_generate_tests(metafunc):
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", metafunc.config.getoption("--images"))
    # if "ec2_instance_type" in metafunc.fixturenames:
    #     metafunc.parametrize("ec2_instance_type", metafunc.config.getoption("--ec2-instance-types"))
