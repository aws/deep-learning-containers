"""Single-node multi-GPU EC2 instance, launched from a capacity reservation.

Multi-GPU tests need several GPUs on one host, and the CI account holds reservations for
several such instance types in differing amounts. Pinning one type makes the test fail for
lack of capacity rather than for a defect in the image, so callers pass a priority-ordered
candidate ladder and the first type with a free reservation wins.

Every rung must carry at least MIN_GPUS GPUs; the tests size themselves from the GPU count
at runtime, so rungs may differ in GPU count and family.
"""

import contextlib
import logging
import os

from botocore.exceptions import ClientError
from test_utils.aws import AWSSessionManager, LoggedConnection
from test_utils.constants import EC2_INSTANCE_ROLE_NAME
from test_utils.efa_helpers import (
    cleanup_stale_instances,
    cleanup_stale_runner_sgs,
    create_runner_ssh_sg,
    delete_runner_ssh_sg,
    get_available_reservations,
    get_default_subnet,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Ordered by reservation headroom, so scarce pools stay free for suites that need them.
DEFAULT_CANDIDATES = ("g6.12xlarge", "g6e.12xlarge", "p4d.24xlarge")

MIN_GPUS = 2

GPU_TEST_TAG_KEY = "dlc-multi-gpu-test"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-west-2")


def candidates_from_env():
    """Candidate ladder from MULTI_GPU_INSTANCE_TYPES (comma-separated), else the default."""
    raw = os.environ.get("MULTI_GPU_INSTANCE_TYPES", "")
    types = [t.strip() for t in raw.split(",") if t.strip()]
    return tuple(types) if types else DEFAULT_CANDIDATES


def _run_params(ami_id, instance_type, key_name, subnet_id, sg_ids, az, cr_id):
    return {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": subnet_id,
        "SecurityGroupIds": list(sg_ids),
        "Placement": {"AvailabilityZone": az},
        "CapacityReservationSpecification": {
            "CapacityReservationTarget": {"CapacityReservationId": cr_id},
        },
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
        },
        "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 300}}],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"CI-CD multi-gpu {instance_type}"},
                    {"Key": GPU_TEST_TAG_KEY, "Value": "true"},
                ],
            }
        ],
        "IamInstanceProfile": {"Name": EC2_INSTANCE_ROLE_NAME},
    }


def launch_from_candidates(aws_session, ami_id, key_name, sg_ids, candidates=None):
    """Launch one instance of the first candidate type with a free reservation.

    Tries every reservation of a type (they differ by AZ) before moving to the next type,
    so a single dry AZ does not skip a type that has capacity elsewhere.

    Returns (instance_id, instance_type). Raises RuntimeError if no candidate has capacity.
    """
    candidates = candidates or candidates_from_env()
    tried = []

    for instance_type in candidates:
        reservations = get_available_reservations(aws_session, instance_type, min_count=1)
        if not reservations:
            LOGGER.info(f"{instance_type}: no reservation with free capacity, next candidate")
            tried.append(f"{instance_type} (no free reservation)")
            continue

        for reservation in reservations:
            az = reservation["AvailabilityZone"]
            cr_id = reservation["CapacityReservationId"]
            try:
                response = aws_session.ec2.run_instances(
                    **_run_params(
                        ami_id,
                        instance_type,
                        key_name,
                        get_default_subnet(aws_session, az),
                        sg_ids,
                        az,
                        cr_id,
                    )
                )
            except ClientError as e:
                LOGGER.warning(f"{instance_type} in {az} via {cr_id} failed: {e}")
                tried.append(f"{instance_type}/{az}")
                continue

            instance_id = response["Instances"][0]["InstanceId"]
            LOGGER.info(f"Launched {instance_type} in {az} via reservation {cr_id}: {instance_id}")
            return instance_id, instance_type

    raise RuntimeError(
        f"No capacity for any of {list(candidates)}. Tried: {tried or 'nothing'}. "
        f"Widen the ladder via MULTI_GPU_INSTANCE_TYPES or raise the reservation."
    )


@contextlib.contextmanager
def gpu_instance(candidates=None, region=DEFAULT_REGION):
    """Launch one multi-GPU instance from a reservation, yield an SSH connection, clean up.

    Yields (conn, instance_type, num_gpus).
    """
    aws_session = AWSSessionManager(region=region)
    ami_id = aws_session.get_latest_ami()

    key_name = key_path = runner_sg_id = instance_id = None
    try:
        key_name, key_path = aws_session.create_key_pair()

        # The finally below is skipped on SIGKILL; only a next-run sweep reclaims those.
        cleanup_stale_instances(aws_session, tag_key=GPU_TEST_TAG_KEY)
        cleanup_stale_runner_sgs(aws_session)

        runner_ip = aws_session.get_codebuild_runner_public_ip()
        vpc_id = aws_session.ec2.describe_vpcs(
            Filters=[{"Name": "is-default", "Values": ["true"]}]
        )["Vpcs"][0]["VpcId"]
        runner_sg_id = create_runner_ssh_sg(aws_session, runner_ip, vpc_id, run_id=key_name)

        instance_id, instance_type = launch_from_candidates(
            aws_session, ami_id, key_name, [runner_sg_id], candidates
        )
        aws_session.wait_for_instance_ready(instance_id)

        public_ip = aws_session.ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][
            0
        ]["Instances"][0]["PublicIpAddress"]
        conn = LoggedConnection(
            host=public_ip,
            user="ec2-user",
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )
        # Non-tty runner: without this, fabric blocks trying to proxy stdin.
        conn.config.run.in_stream = False

        num_gpus = int(conn.run("nvidia-smi -L | wc -l", hide=True).stdout.strip())
        if num_gpus < MIN_GPUS:
            raise RuntimeError(
                f"{instance_type} reported {num_gpus} GPU(s); multi-GPU tests need "
                f">= {MIN_GPUS}. Remove it from the candidate ladder."
            )
        LOGGER.info(f"{instance_type} ready with {num_gpus} GPUs")

        yield conn, instance_type, num_gpus
    finally:
        if instance_id:
            with contextlib.suppress(Exception):
                aws_session.ec2.terminate_instances(InstanceIds=[instance_id])
                LOGGER.info(f"Terminated {instance_id}")
        if runner_sg_id:
            with contextlib.suppress(Exception):
                delete_runner_ssh_sg(aws_session, runner_sg_id)
        if key_name:
            with contextlib.suppress(Exception):
                aws_session.delete_key_pair(key_name, key_path)
