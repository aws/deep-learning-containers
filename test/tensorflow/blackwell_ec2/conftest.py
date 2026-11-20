"""EC2 G7 (Blackwell, sm_120) host for the TF training image's GPU tests.

Launches one g7.2xlarge for the session: first into the CI account's targeted
G7 capacity reservation (BLACKWELL_ODCR_ID), and if that has no free instance
(for example a SageMaker endpoint test holds it), from the on-demand pool. The
instance is terminated, and its key pair and security group deleted, at the end.
Run by the non-blocking single-gpu-test-blackwell job
(tensorflow-training.tests-single-gpu.yml).
"""

import logging
import os

import pytest
from botocore.exceptions import ClientError
from test_utils.constants import EC2_INSTANCE_ROLE_NAME

LOGGER = logging.getLogger(__name__)

INSTANCE_TYPE = "g7.2xlarge"
AMI_SSM_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/base-with-single-cuda-amazon-linux-2023/latest/ami-id"
)
# EC2 reports a reservation or pool without free capacity with these codes.
NO_CAPACITY_CODES = (
    "InsufficientInstanceCapacity",
    "InsufficientCapacity",
    "ReservationCapacityExceeded",
    "InsufficientReservedInstanceCapacity",
)


def _run_instance(ec2, base_params, odcr_id):
    """Launch into the reservation if it has room, else on demand. Returns (instance_id, source)."""
    if odcr_id:
        reservation = ec2.describe_capacity_reservations(CapacityReservationIds=[odcr_id])[
            "CapacityReservations"
        ][0]
        subnets = ec2.describe_subnets(
            Filters=[
                {"Name": "availability-zone-id", "Values": [reservation["AvailabilityZoneId"]]},
                {"Name": "default-for-az", "Values": ["true"]},
            ]
        )["Subnets"]
        LOGGER.info(
            f"[capacity] {odcr_id}: {reservation['AvailableInstanceCount']}/"
            f"{reservation['TotalInstanceCount']} free in {reservation['AvailabilityZoneId']}"
        )
        if subnets:
            params = dict(
                base_params,
                SubnetId=subnets[0]["SubnetId"],
                CapacityReservationSpecification={
                    "CapacityReservationTarget": {"CapacityReservationId": odcr_id}
                },
            )
            try:
                response = ec2.run_instances(**params)
                return response["Instances"][0]["InstanceId"], f"reservation {odcr_id}"
            except ClientError as e:
                if e.response["Error"]["Code"] not in NO_CAPACITY_CODES:
                    raise
                LOGGER.warning(f"[capacity] {odcr_id} has no free instance: {e}; using on-demand")
    response = ec2.run_instances(
        **base_params, CapacityReservationSpecification={"CapacityReservationPreference": "none"}
    )
    return response["Instances"][0]["InstanceId"], "on-demand"


@pytest.fixture(scope="session")
def g7_instance(aws_session):
    """Yield (instance_id, key_path) for a running g7.2xlarge; clean everything up after."""
    ec2 = aws_session.ec2
    ami_id = aws_session.get_latest_ami(parameter=AMI_SSM_PARAMETER)
    key_name, key_path, sg_id, instance_id = None, None, None, None
    try:
        key_name, key_path = aws_session.create_key_pair()
        sg_id = aws_session.create_ssh_security_group()
        base_params = {
            "ImageId": ami_id,
            "InstanceType": INSTANCE_TYPE,
            "KeyName": key_name,
            "MinCount": 1,
            "MaxCount": 1,
            "SecurityGroupIds": [sg_id],
            "IamInstanceProfile": {"Name": EC2_INSTANCE_ROLE_NAME},
            "MetadataOptions": {
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 2,
            },
            "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 150}}],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": "CI-CD tf-blackwell-ec2-test"}],
                }
            ],
        }
        instance_id, source = _run_instance(
            ec2, base_params, os.environ.get("BLACKWELL_ODCR_ID", "").strip()
        )
        LOGGER.info(
            f"[capacity] launched {instance_id} ({INSTANCE_TYPE}, ami {ami_id}) from {source}"
        )
        aws_session.wait_for_instance_ready(instance_id)
        yield instance_id, key_path
    finally:
        if instance_id:
            aws_session.terminate_instance(instance_id)
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        if sg_id:
            aws_session.delete_security_group(sg_id)
        if key_name:
            aws_session.delete_key_pair(key_name, key_path)


@pytest.fixture(scope="session")
def conn(aws_session, g7_instance):
    instance_id, key_path = g7_instance
    return aws_session.get_ssh_connection(instance_id, key_path)
