"""Telemetry test configuration and fixtures."""

import logging

import pytest

LOGGER = logging.getLogger(__name__)

TELEMETRY_INSTANCE_TYPE = {"x86": "m5.xlarge", "arm64": "m7g.xlarge"}
AMI_SSM_PARAMETER = {
    "x86": "/aws/service/deeplearning/ami/x86_64/base-with-single-cuda-amazon-linux-2023/latest/ami-id",
    "arm64": "/aws/service/deeplearning/ami/arm64/base-with-single-cuda-amazon-linux-2023/latest/ami-id",
}
DOCKER_RUN = "docker run -d -it --rm"
DOCKER_EXEC = "docker exec"
DOCKER_RM = "docker rm -f"


def pytest_addoption(parser):
    parser.addoption("--framework", required=True)
    parser.addoption("--framework-version", required=True)
    parser.addoption("--container-type", required=True)
    parser.addoption("--arch-type", default="x86")


@pytest.fixture(scope="session")
def framework(request):
    return request.config.getoption("--framework")


@pytest.fixture(scope="session")
def framework_version(request):
    return request.config.getoption("--framework-version")


@pytest.fixture(scope="session")
def container_type(request):
    return request.config.getoption("--container-type")


@pytest.fixture(scope="session")
def arch_type(request):
    return request.config.getoption("--arch-type")


@pytest.fixture(scope="session")
def ec2_instance(request, aws_session, arch_type):
    """Launch an EC2 instance (arch-matched to the image) for the session, tear down after."""
    ami_id = aws_session.get_latest_ami(parameter=AMI_SSM_PARAMETER[arch_type])
    instance_type = TELEMETRY_INSTANCE_TYPE[arch_type]
    LOGGER.info(f"Setting up EC2 instance: arch={arch_type}, ami={ami_id}, type={instance_type}")

    key_name, key_path = None, None
    instance_id = None
    sg_id = None
    try:
        key_name, key_path = aws_session.create_key_pair()
        sg_id = aws_session.create_ssh_security_group()
        instance_id = aws_session.launch_instance(
            ami_id=ami_id,
            instance_type=instance_type,
            key_name=key_name,
            instance_name="telemetry-test",
            security_group_ids=[sg_id],
        )
        aws_session.wait_for_instance_ready(instance_id)
        yield instance_id, key_path
    finally:
        if instance_id:
            aws_session.terminate_instance(instance_id)
            # Wait for instance to terminate before deleting SG
            aws_session.ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        if sg_id:
            aws_session.delete_security_group(sg_id)
        if key_name:
            aws_session.delete_key_pair(key_name, key_path)


@pytest.fixture(scope="session")
def conn(aws_session, ec2_instance):
    """SSH connection to the EC2 instance."""
    instance_id, key_path = ec2_instance
    LOGGER.info(f"Establishing SSH connection to {instance_id}")
    return aws_session.get_ssh_connection(instance_id, key_path)


@pytest.fixture(scope="session")
def pull_image(conn, image_uri, region):
    """Authenticate ECR and pull the image once per session."""
    LOGGER.info(f"Pulling image {image_uri}")
    account_id = image_uri.split(".")[0]
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    conn.run(
        f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry}",
    )
    conn.run(f"docker pull {image_uri}")
