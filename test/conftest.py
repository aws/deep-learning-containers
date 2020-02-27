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
        "--ec2-instance-types",
        required=False,
        nargs='+',
        help="Specify image(s) to run"
    )


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


@pytest.fixture(scope="session")
def ec2_client():
    return boto3.client('ec2')


@pytest.mark.timeout(600)
@pytest.fixture(scope="session")
def start_ec2_instance(ec2_client, ec2_instance_type):
    key = ec2_client.create_key_pair(KeyName="pytest.pem")
    instances = ec2_client.create_instances(
        KeyName=key.get('KeyName'),
        ImageId='ami-025ed45832b817a35',
        InstanceType=ec2_instance_type,
        MaxCount=1,
        MinCount=1
    )
    instance_id = instances[0].id




@pytest.fixture(scope="session")
def run_on_ec2_instance(request, start_ec2_instance):
    ec2 = boto3.client('ec2')
    return request.config.getoption("--ec2-instance-type")


def pytest_generate_tests(metafunc):
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", metafunc.config.getoption("--images"))
    if "ec2_instance_type" in metafunc.fixturenames:
        metafunc.parametrize("ec2_instance_type", metafunc.config.getoption("--ec2-instance-types"))
