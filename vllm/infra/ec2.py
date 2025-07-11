import os
import time
import re
import logging
import sys
import uuid
import boto3
from contextlib import contextmanager

import test.test_utils.ec2 as ec2_utils
from infra.utils.fsx_utils import FsxSetup

from botocore.config import Config
from fabric import Connection


from test import test_utils
from test.test_utils import KEYS_TO_DESTROY_FILE

from test.test_utils.ec2 import get_default_vpc_id, get_default_subnet_for_az, get_subnet_id_by_vpc

# Constant to represent default region for boto3 commands
DEFAULT_REGION = "us-west-2"
# Constant to represent region where p4de tests can be run
P4DE_REGION = "us-east-1"

EC2_INSTANCE_ROLE_NAME = "ec2TestInstanceRole"

VLLM_INSTANCE_TYPE = ["p4d.24xlarge", "p5.48xlarge"]

ENABLE_IPV6_TESTING = os.getenv("ENABLE_IPV6_TESTING", "false").lower() == "true"


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


def ec2_instance_ami(region):
    return test_utils.get_dlami_id(region)


def availability_zone_options(ec2_client, ec2_instance_type, region):
    """
    Parametrize with a reduced list of availability zones for particular instance types for which
    capacity has been reserved in that AZ. For other instance types, parametrize with list of all
    AZs in the region.
    :param ec2_client: boto3 Client for EC2
    :param ec2_instance_type: str instance type for which AZs must be determined
    :param region: str region in which instance must be created
    :return: list of str AZ names
    """
    allowed_availability_zones = None
    if ec2_instance_type in ["p4de.24xlarge"]:
        if region == "us-east-1":
            allowed_availability_zones = ["us-east-1d", "us-east-1c"]
    if ec2_instance_type in ["p4d.24xlarge"]:
        if region == "us-west-2":
            allowed_availability_zones = ["us-west-2b", "us-west-2c"]
    if not allowed_availability_zones:
        allowed_availability_zones = ec2_utils.get_availability_zone_ids(ec2_client)
    return allowed_availability_zones


def efa_ec2_instances(
    ec2_client,
    ec2_instance_type,
    ec2_instance_role_name,
    ec2_key_name,
    ec2_instance_ami,
    region,
    availability_zone_options,
):
    ec2_key_name = f"{ec2_key_name}-{str(uuid.uuid4())}"
    print(f"Creating instance: CI-CD {ec2_key_name}")
    key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
    print(f"Using AMI for EFA EC2 {ec2_instance_ami}")

    def delete_ssh_keypair():
        if test_utils.is_pr_context():
            test_utils.destroy_ssh_keypair(ec2_client, key_filename)
        else:
            with open(KEYS_TO_DESTROY_FILE, "a") as destroy_keys:
                destroy_keys.write(f"{key_filename}\n")

    volume_name = "/dev/sda1" if ec2_instance_ami in test_utils.UL_AMI_LIST else "/dev/xvda"

    instance_name_prefix = f"CI-CD {ec2_key_name}"
    ec2_run_instances_definition = {
        "BlockDeviceMappings": [
            {
                "DeviceName": volume_name,
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": 150,
                    "VolumeType": "gp3",
                    "Iops": 3000,
                    "Throughput": 125,
                },
            },
        ],
        "ImageId": ec2_instance_ami,
        "InstanceType": ec2_instance_type,
        "IamInstanceProfile": {"Name": ec2_instance_role_name},
        "KeyName": ec2_key_name,
        "MaxCount": 2,
        "MinCount": 2,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": instance_name_prefix}]}
        ],
    }
    instances = ec2_utils.launch_efa_instances_with_retry(
        ec2_client,
        ec2_instance_type,
        availability_zone_options,
        ec2_run_instances_definition,
    )

    def terminate_efa_instances():
        ec2_client.terminate_instances(
            InstanceIds=[instance_info["InstanceId"] for instance_info in instances]
        )

    master_instance_id = instances[0]["InstanceId"]
    ec2_utils.check_instance_state(master_instance_id, state="running", region=region)
    ec2_utils.check_system_state(
        master_instance_id, system_status="ok", instance_status="ok", region=region
    )
    print(f"Master instance {master_instance_id} is ready")

    if len(instances) > 1:
        ec2_utils.create_name_tags_for_instance(
            master_instance_id, f"{instance_name_prefix}_master", region
        )
        for i in range(1, len(instances)):
            worker_instance_id = instances[i]["InstanceId"]
            ec2_utils.create_name_tags_for_instance(
                worker_instance_id, f"{instance_name_prefix}_worker_{i}", region
            )
            ec2_utils.check_instance_state(worker_instance_id, state="running", region=region)
            ec2_utils.check_system_state(
                worker_instance_id, system_status="ok", instance_status="ok", region=region
            )
            print(f"Worker instance {worker_instance_id} is ready")

    num_efa_interfaces = ec2_utils.get_num_efa_interfaces_for_instance_type(
        ec2_instance_type, region=region
    )
    if num_efa_interfaces > 1:
        # p4d instances require attaching elastic ip to connect to them
        elastic_ip_allocation_ids = []
        # create and attach network interfaces and elastic ips to all instances
        for instance in instances:
            instance_id = instance["InstanceId"]

            network_interface_id = ec2_utils.get_network_interface_id(instance_id, region)

            elastic_ip_allocation_id = ec2_utils.attach_elastic_ip(
                network_interface_id, region, ENABLE_IPV6_TESTING
            )
            elastic_ip_allocation_ids.append(elastic_ip_allocation_id)

        def elastic_ips_finalizer():
            ec2_utils.delete_elastic_ips(elastic_ip_allocation_ids, ec2_client)

    return_val = [(instance_info["InstanceId"], key_filename) for instance_info in instances]
    print(f"Launched EFA Test instances - {[instance_id for instance_id, _ in return_val]}")

    return return_val


@contextmanager
def ec2_test_environment():
    cleanup_functions = []
    try:
        # Setup code here
        region = DEFAULT_REGION
        ec2_cli = ec2_client(region)
        instance_type = VLLM_INSTANCE_TYPE[0]
        ami_id = ec2_instance_ami(region)
        az_options = availability_zone_options(ec2_cli, instance_type, region)

        instances_info = efa_ec2_instances(
            ec2_client=ec2_cli,
            ec2_instance_type=instance_type,
            ec2_instance_role_name=EC2_INSTANCE_ROLE_NAME,
            ec2_key_name="vllm-ec2-test",
            ec2_instance_ami=ami_id,
            region=region,
            availability_zone_options=az_options,
        )
        # Register cleanup functions
        cleanup_functions.extend(
            [
                lambda: ec2_cli.terminate_instances(
                    InstanceIds=[instance_id for instance_id, _ in instances_info]
                ),
                lambda: test_utils.destroy_ssh_keypair(ec2_cli, instances_info[0][1]),
            ]
        )

        yield instances_info

    finally:
        print("Running cleanup operations...")
        for cleanup_func in cleanup_functions:
            try:
                if cleanup_func is not None:
                    cleanup_func()
            except Exception as cleanup_error:
                LOGGER.error(f"Error during cleanup: {str(cleanup_error)}")


def setup():
    print("Testing vllm on ec2........")
    # with ec2_test_environment() as instances_info:
    #     print("Test setup is completed")
    fsx = FsxSetup(DEFAULT_REGION)
    ec2_cli = ec2_client(DEFAULT_REGION)

    vpc_id = get_default_vpc_id(ec2_cli)
    subnet_id = get_subnet_id_by_vpc(ec2_cli, vpc_id)

    # create fsx
    sg_fsx = fsx.create_security_group(vpc_id, "vllm-ec2-fsx-sg", "SG for Fsx Mounting")
    ingress_rules = {"protocol": "tcp", "port": "988-1023"}
    fsx.add_security_group_ingress_rules(sg_fsx, ingress_rules)

    fsx_config = fsx.create_fsx_filesystem(
        subnet_id, sg_fsx, 1200, "SCRATCH_2", {"Name": "vllm-fsx-storage"}
    )

    print(fsx_config)


if __name__ == "__main__":
    setup()
