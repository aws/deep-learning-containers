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


def _setup_instance_async(connection, fsx_dns_name, mount_name):
    """
    Setup FSx mount and VLLM environment on an instance asynchronously
    """
    # Copy script to instance
    connection.put("vllm/infra/utils/setup_fsx_vllm.sh", "/home/ec2-user/setup_fsx_vllm.sh")

    # Make script executable and run it
    commands = [
        "chmod +x /home/ec2-user/setup_fsx_vllm.sh",
        f"/home/ec2-user/setup_fsx_vllm.sh {fsx_dns_name} {mount_name}",
    ]

    # Execute commands asynchronously
    promise = connection.run("; ".join(commands), asynchronous=True)
    return promise


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
            fsx.delete_security_group(sg_fsx)
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


def setup():
    print("Testing vllm on ec2........")
    fsx = FsxSetup(DEFAULT_REGION)
    ec2_cli = ec2_client(DEFAULT_REGION)
    sg_fsx = None
    fsx_config = None
    instances_info = None

    try:
        vpc_id = get_default_vpc_id(ec2_cli)
        subnet_ids = get_subnet_id_by_vpc(ec2_cli, vpc_id)

        # Create security group
        try:
            sg_fsx = fsx.create_security_group(vpc_id, "vllm-ec2-fsx-sg", "SG for Fsx Mounting")
            fsx.add_security_group_ingress_and_egress_rules(sg_fsx)
            print(f"Created security group: {sg_fsx}")
        except Exception as e:
            print(f"Error creating security group: {str(e)}")
            cleanup_resources(ec2_cli, None, sg_fsx, None, fsx)
            raise

        # Create FSx filesystem
        try:
            fsx_config = fsx.create_fsx_filesystem(
                subnet_ids[0], [sg_fsx], 1200, "SCRATCH_2", {"Name": "vllm-fsx-storage"}
            )
            print(f"Created FSx filesystem: {fsx_config}")
        except Exception as e:
            print(f"Error creating FSx filesystem: {str(e)}")
            cleanup_resources(ec2_cli, None, sg_fsx, fsx_config, fsx)
            raise

        # Launch EC2 instances
        try:
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
        except Exception as e:
            print(f"Error launching EC2 instances: {str(e)}")
            cleanup_resources(ec2_cli, instances_info, sg_fsx, fsx_config, fsx)
            raise

        # Wait for instances to initialize
        time.sleep(60)

        # Setup FSx on instances
        setup_promises = []
        try:
            for instance_id, key_filename in instances_info:
                instance_details = ec2_cli.describe_instances(InstanceIds=[instance_id])[
                    "Reservations"
                ][0]["Instances"][0]
                public_ip = instance_details.get("PublicIpAddress")

                if not public_ip:
                    raise Exception(f"No public IP found for instance {instance_id}")

                connection = Connection(
                    host=public_ip,
                    user="ec2-user",
                    connect_kwargs={
                        "key_filename": key_filename,
                    },
                )

                promise = _setup_instance_async(
                    connection, fsx_config["dns_name"], fsx_config["mount_name"]
                )
                setup_promises.append((instance_id, promise))

            # Wait for all setups to complete
            for instance_id, promise in setup_promises:
                try:
                    promise.join()
                    print(f"Setup completed successfully for instance {instance_id}")
                except Exception as e:
                    print(f"Error setting up instance {instance_id}: {str(e)}")
                    raise

        except Exception as e:
            print(f"Error during instance setup: {str(e)}")
            cleanup_resources(ec2_cli, instances_info, sg_fsx, fsx_config, fsx)
            raise

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        # Final cleanup attempt if not already done
        cleanup_resources(ec2_cli, instances_info, sg_fsx, fsx_config, fsx)
        raise

    finally:
        # Cleanup all resources in success case or if previous cleanup attempts failed
        cleanup_resources(ec2_cli, instances_info, sg_fsx, fsx_config, fsx)


if __name__ == "__main__":
    setup()
