# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Telemetry test configuration and fixtures."""

import logging

import pytest

LOGGER = logging.getLogger(__name__)

TELEMETRY_INSTANCE_TYPE = "m5.xlarge"
DOCKER_RUN = "docker run -d -it --rm"
DOCKER_EXEC = "docker exec"
DOCKER_RM = "docker rm -f"


def pytest_addoption(parser):
    parser.addoption("--framework", required=True)
    parser.addoption("--framework-version", required=True)
    parser.addoption("--container-type", required=True)


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
def ec2_instance(request, aws_session):
    """Launch an EC2 instance for the test session, tear down after."""
    ami_id = aws_session.get_latest_ami()
    LOGGER.info(f"Setting up EC2 instance: ami={ami_id}, type={TELEMETRY_INSTANCE_TYPE}")

    key_name, key_path = None, None
    instance_id = None
    sg_id = None
    try:
        key_name, key_path = aws_session.create_key_pair()
        sg_id = aws_session.create_ssh_security_group()
        instance_id = aws_session.launch_instance(
            ami_id=ami_id,
            instance_type=TELEMETRY_INSTANCE_TYPE,
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
