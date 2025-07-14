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
from concurrent.futures import ThreadPoolExecutor

from botocore.config import Config
from fabric import Connection


from test import test_utils
from test.test_utils import KEYS_TO_DESTROY_FILE

from test.test_utils.ec2 import (
    get_default_vpc_id,
    get_subnet_id_by_vpc,
)

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


def _setup_instance(connection, fsx_dns_name, mount_name):
    """
    Setup FSx mount and VLLM environment on an instance synchronously
    """
    # Copy script to instance
    connection.put("vllm/infra/utils/setup_fsx_vllm.sh", "/home/ec2-user/setup_fsx_vllm.sh")

    # Make script executable and run it
    commands = [
        "chmod +x /home/ec2-user/setup_fsx_vllm.sh",
        f"/home/ec2-user/setup_fsx_vllm.sh {fsx_dns_name} {mount_name}",
    ]

    # Execute commands synchronously
    result = connection.run("; ".join(commands))
    return result


def cleanup_resources(ec2_cli, instances_info=None, sg_fsx=None, fsx_config=None, fsx=None):
    """
    Cleanup all resources in reverse order of creation
    """
    cleanup_errors = []

    # Cleanup instances if they exist
    if instances_info:
        try:
            instance_ids = [instance_id for instance_id, _ in instances_info]
            ec2_cli.terminate_instances(InstanceIds=instance_ids)
            print(f"Terminated EC2 instances: {instance_ids}")

            # Wait for instances to terminate
            waiter = ec2_cli.get_waiter("instance_terminated")
            waiter.wait(InstanceIds=instance_ids)
        except Exception as e:
            cleanup_errors.append(f"Failed to terminate EC2 instances: {str(e)}")

    # Cleanup security group if it exists
    if sg_fsx and fsx:
        try:
            fsx.delete_security_group(ec2_cli, sg_fsx)
            print(f"Deleted security group: {sg_fsx}")
        except Exception as e:
            cleanup_errors.append(f"Failed to delete security group: {str(e)}")

    # Cleanup FSx filesystem if it exists
    if fsx_config and fsx:
        try:
            fsx.delete_fsx_filesystem(fsx_config["filesystem_id"])
            print(f"Deleted FSx filesystem: {fsx_config['filesystem_id']}")
        except Exception as e:
            cleanup_errors.append(f"Failed to delete FSx filesystem: {str(e)}")

    if cleanup_errors:
        error_message = "\n".join(cleanup_errors)
        raise Exception(f"Cleanup errors occurred:\n{error_message}")


def launch_ec2_instances(ec2_cli):
    """Launch EC2 instances with EFA support"""
    instance_type = VLLM_INSTANCE_TYPE[0]
    ami_id = ec2_instance_ami(DEFAULT_REGION)
    az_options = availability_zone_options(ec2_cli, instance_type, DEFAULT_REGION)

    instances_info = efa_ec2_instances(
        ec2_client=ec2_cli,
        ec2_instance_type=instance_type,
        ec2_instance_role_name=EC2_INSTANCE_ROLE_NAME,
        ec2_key_name="vllm-ec2-test",
        ec2_instance_ami=ami_id,
        region=DEFAULT_REGION,
        availability_zone_options=az_options,
    )
    print(f"Launched instances: {instances_info}")
    return instances_info


def configure_security_groups(ec2_cli, fsx, vpc_id, instances_info):
    """
    Configure security groups for FSx and EC2 instances

    Args:
        ec2_cli: boto3 EC2 client
        fsx: FsxSetup instance
        vpc_id: VPC ID where security group will be created
        instances_info: List of tuples containing (instance_id, key_filename)

    Returns:
        str: FSx security group ID
    """
    try:
        # Create FSx security group
        sg_fsx = fsx.create_fsx_security_group(
            ec2_cli,
            vpc_id,
            "fsx-lustre-sg-vllm-ec2-tests",
            "Security group for FSx Lustre VLLM EC2 Tests",
        )
        print(f"Created FSx security group: {sg_fsx}")

        # Get instance IDs from instances_info
        instance_ids = [instance_id for instance_id, _ in instances_info]

        # Add security group rules
        fsx.add_ingress_rules_sg(ec2_cli, sg_fsx, instance_ids)

        return sg_fsx

    except Exception as e:
        print(f"Error configuring security groups: {str(e)}")
        raise


def setup_instance(instance_id, key_filename, ec2_cli, fsx_dns_name, mount_name):
    """Setup FSx mount on a single instance"""
    instance_details = ec2_cli.describe_instances(InstanceIds=[instance_id])["Reservations"][0][
        "Instances"
    ][0]
    public_ip = instance_details.get("PublicIpAddress")

    if not public_ip:
        raise Exception(f"No public IP found for instance {instance_id}")

    connection = Connection(
        host=public_ip,
        user="ec2-user",
        connect_kwargs={"key_filename": key_filename},
    )

    return _setup_instance(connection, fsx_dns_name, mount_name)


def setup():
    """Main setup function for VLLM on EC2 with FSx"""
    print("Testing vllm on ec2........")
    fsx = FsxSetup(DEFAULT_REGION)
    ec2_cli = ec2_client(DEFAULT_REGION)
    resources = {"instances_info": None, "sg_fsx": None, "fsx_config": None}

    try:
        # Get VPC and subnet information
        vpc_id = get_default_vpc_id(ec2_cli)
        subnet_ids = get_subnet_id_by_vpc(ec2_cli, vpc_id)

        # Launch EC2 instances
        resources["instances_info"] = launch_ec2_instances(ec2_cli)
        time.sleep(60)  # Wait for instances to initialize

        # Configure security groups
        resources["sg_fsx"] = configure_security_groups(
            ec2_cli, fsx, vpc_id, resources["instances_info"]
        )

        # Create FSx filesystem
        resources["fsx_config"] = fsx.create_fsx_filesystem(
            subnet_ids[0], [resources["sg_fsx"]], 1200, "SCRATCH_2", {"Name": "vllm-fsx-storage"}
        )
        print(f"Created FSx filesystem: {resources['fsx_config']}")

        with ThreadPoolExecutor(max_workers=len(resources["instances_info"])) as executor:
            future_to_instance = {
                executor.submit(
                    setup_instance,
                    instance_id,
                    key_filename,
                    ec2_cli,
                    resources["fsx_config"]["dns_name"],
                    resources["fsx_config"]["mount_name"],
                ): instance_id
                for instance_id, key_filename in resources["instances_info"]
            }

            for future in future_to_instance:
                instance_id = future_to_instance[future]
                try:
                    future.result()
                    print(f"Setup completed successfully for instance {instance_id}")
                except Exception as e:
                    raise Exception(f"Error setting up instance {instance_id}: {str(e)}")

        return resources
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        cleanup_resources(
            ec2_cli, resources["instances_info"], resources["sg_fsx"], resources["fsx_config"], fsx
        )
        raise


if __name__ == "__main__":
    setup()
