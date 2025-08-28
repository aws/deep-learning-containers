import os
import time
import re
import logging
import sys
import uuid
import boto3
from contextlib import contextmanager


from test import test_utils
from test.test_utils import DEFAULT_REGION, P4DE_REGION
import test.test_utils.ec2 as ec2_utils
from test.vllm.ec2.utils.fsx_utils import FsxSetup
from concurrent.futures import ThreadPoolExecutor

from botocore.config import Config
from fabric import Connection
from botocore.exceptions import ClientError, WaiterError


from test.test_utils import KEYS_TO_DESTROY_FILE, AL2023_BASE_DLAMI_ARM64_US_WEST_2

from test.test_utils.ec2 import (
    get_default_vpc_id,
    get_subnet_id_by_vpc,
    get_efa_ec2_instance_type,
    get_ec2_instance_type,
    filter_efa_only_p4_instance_type,
)

# Constant to represent default region for boto3 commands
DEFAULT_REGION = "us-west-2"
EC2_INSTANCE_ROLE_NAME = "ec2TestInstanceRole"
ENABLE_IPV6_TESTING = os.getenv("ENABLE_IPV6_TESTING", "false").lower() == "true"


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

TEST_ID = str(uuid.uuid4())


def ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


def ec2_instance_ami(region, image):
    if "arm64" in image:
        return AL2023_BASE_DLAMI_ARM64_US_WEST_2

    return test_utils.get_dlami_id(region)


def ec2_instance_type(image):
    if "arm64" in image:
        return "g5g.16xlarge"
    else:
        return "p4d.24xlarge"


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


def check_ip_rule_exists(security_group_rules, ip_address):
    """
    Check if an IP rule exists in security group rules
    """
    if not security_group_rules:
        return False

    for rule in security_group_rules:
        if (
            rule.get("FromPort") == 80
            and rule.get("ToPort") == 80
            and rule.get("IpProtocol") == "tcp"
            and "IpRanges" in rule
        ):
            for ip_range in rule.get("IpRanges", []):
                if ip_range.get("CidrIp") == f"{ip_address}/32":
                    LOGGER.info(f"Found existing rule for IP {ip_address}")
                    return True
    return False


def authorize_ingress(ec2_client, group_id, ip_address):
    try:
        response = ec2_client.describe_security_groups(GroupIds=[group_id])
        if response.get("SecurityGroups") and response["SecurityGroups"]:
            existing_rules = response["SecurityGroups"][0].get("IpPermissions", [])
            if check_ip_rule_exists(existing_rules, ip_address):
                LOGGER.info("Ingress rule already exists, skipping creation.")
                return

        ec2_client.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 8000,
                    "ToPort": 8000,
                    "IpRanges": [
                        {
                            "CidrIp": f"{ip_address}/32",
                            "Description": "Temporary access for vLLM testing",
                        }
                    ],
                }
            ],
        )
        LOGGER.info("Ingress rule added successfully.")
    except ClientError as e:
        LOGGER.error(f"Failed to authorize ingress: {str(e)}")
        raise


def setup_test_artifacts(ec2_client, instances, key_filename, region):
    ec2_connections = {}
    master_connection = None
    worker_connection = None

    for instance in instances:
        instance_id = instance["InstanceId"]
        try:
            instance_details = ec2_client.describe_instances(InstanceIds=[instance_id])[
                "Reservations"
            ][0]["Instances"][0]
            public_ip = instance_details.get("PublicIpAddress")

            if not public_ip:
                raise Exception(f"No public IP found for instance {instance_id}")

            connection = Connection(
                host=public_ip,
                user="ec2-user",
                connect_kwargs={"key_filename": key_filename},
            )

            # Test connection
            connection.run('echo "Connection test"', hide=True)
            ec2_connections[instance_id] = connection

            if not master_connection:
                master_connection = connection
            else:
                worker_connection = connection

            print(f"Successfully connected to instance {instance_id}")

        except Exception as e:
            print(f"Failed to connect to instance {instance_id}: {str(e)}")
            raise

    artifact_folder = f"vllm-{TEST_ID}-folder"
    s3_test_artifact_location = test_utils.upload_tests_to_s3(artifact_folder)

    def delete_s3_artifact_copy():
        test_utils.delete_uploaded_tests_from_s3(s3_test_artifact_location)

    # Setup master instance
    master_connection.run("rm -rf $HOME/container_tests")
    master_connection.run(
        f"aws s3 cp --recursive {test_utils.TEST_TRANSFER_S3_BUCKET}/{artifact_folder} $HOME/container_tests --region {test_utils.TEST_TRANSFER_S3_BUCKET_REGION}"
    )
    print(f"Successfully copying {test_utils.TEST_TRANSFER_S3_BUCKET} for master")
    master_connection.run(
        f"mkdir -p $HOME/container_tests/logs && chmod -R +x $HOME/container_tests/*"
    )

    worker_connection.run("rm -rf $HOME/container_tests")
    worker_connection.run(
        f"aws s3 cp --recursive {test_utils.TEST_TRANSFER_S3_BUCKET}/{artifact_folder} $HOME/container_tests --region {test_utils.TEST_TRANSFER_S3_BUCKET_REGION}"
    )
    print(f"Successfully copying {test_utils.TEST_TRANSFER_S3_BUCKET} for worker")
    worker_connection.run(
        f"mkdir -p $HOME/container_tests/logs && chmod -R +x $HOME/container_tests/*"
    )

    # Cleanup S3 artifacts
    delete_s3_artifact_copy()

    return [master_connection, worker_connection]


def launch_regular_instances_with_retry(
    ec2_client,
    ec2_instance_type,
    availability_zone_options,
    ec2_run_instances_definition,
):
    """
    Launch regular (non-EFA) EC2 instances with retry capability
    """
    instances = None
    error = None

    for a_zone in availability_zone_options:
        ec2_run_instances_definition["Placement"] = {"AvailabilityZone": a_zone}
        try:
            instances = ec2_client.run_instances(**ec2_run_instances_definition)["Instances"]
            if instances:
                break
        except ClientError as e:
            LOGGER.error(f"Failed to launch in {a_zone} due to {e}")
            error = e
            continue

    if not instances:
        raise error or Exception("Failed to launch instances in any availability zone")

    return instances


def efa_ec2_instances(
    ec2_client,
    ec2_instance_type,
    ec2_instance_role_name,
    ec2_key_name,
    ec2_instance_ami,
    region,
    availability_zone_options,
    is_arm64,
):
    instances = None
    key_filename = None
    elastic_ip_allocation_ids = []
    is_efa = not is_arm64

    try:
        ec2_key_name = f"{ec2_key_name}-{TEST_ID}"
        print(f"Creating instance: CI-CD {ec2_key_name}")
        key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
        volume_name = "/dev/sda1" if ec2_instance_ami in test_utils.UL_AMI_LIST else "/dev/xvda"

        instance_name_prefix = f"CI-CD {ec2_key_name}"
        ec2_run_instances_definition = {
            "BlockDeviceMappings": [
                {
                    "DeviceName": volume_name,
                    "Ebs": {
                        "DeleteOnTermination": True,
                        "VolumeSize": 600,
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
            "MaxCount": 1 if is_efa else 2,
            "MinCount": 1 if is_efa else 2,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": instance_name_prefix}],
                }
            ],
        }

        if is_efa:
            instances = ec2_utils.launch_efa_instances_with_retry(
                ec2_client,
                ec2_instance_type,
                availability_zone_options,
                ec2_run_instances_definition,
            )
        else:
            instances = launch_regular_instances_with_retry(
                ec2_client,
                ec2_instance_type,
                availability_zone_options,
                ec2_run_instances_definition,
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
            for instance in instances:
                try:
                    instance_id = instance["InstanceId"]

                    network_interface_id = ec2_utils.get_network_interface_id(instance_id, region)
                    elastic_ip_allocation_id = ec2_utils.attach_elastic_ip(
                        network_interface_id, region, ENABLE_IPV6_TESTING
                    )
                    elastic_ip_allocation_ids.append(elastic_ip_allocation_id)
                except Exception as e:
                    if elastic_ip_allocation_ids:
                        ec2_utils.delete_elastic_ips(elastic_ip_allocation_ids, ec2_client)
                    raise Exception(f"Error allocating elastic IP: {str(e)}")

        connections = setup_test_artifacts(ec2_client, instances, key_filename, region)
        return_val = {
            "instances": [
                (instance_info["InstanceId"], key_filename) for instance_info in instances
            ],
            "elastic_ips": elastic_ip_allocation_ids,
            "connections": connections,
        }
        print("Launched EFA Test instances")
        return return_val

    except Exception as e:
        print(f"Error in efa_ec2_instances: {str(e)}")
        # Clean up elastic IPs
        if elastic_ip_allocation_ids:
            try:
                ec2_utils.delete_elastic_ips(elastic_ip_allocation_ids, ec2_client)
            except Exception as cleanup_error:
                print(f"Error cleaning up elastic IPs: {str(cleanup_error)}")

        # Clean up instances
        if instances:
            try:
                instance_ids = [instance["InstanceId"] for instance in instances]
                ec2_client.terminate_instances(InstanceIds=instance_ids)
                # Wait for instances to terminate
                waiter = ec2_client.get_waiter("instance_terminated")
                waiter.wait(InstanceIds=instance_ids)
            except Exception as cleanup_error:
                print(f"Error terminating instances: {str(cleanup_error)}")

        # Clean up key pair
        if key_filename:
            try:
                if os.path.exists(key_filename):
                    os.remove(key_filename)
                if os.path.exists(f"{key_filename}.pub"):
                    os.remove(f"{key_filename}.pub")
            except Exception as cleanup_error:
                print(f"Error cleaning up key files: {str(cleanup_error)}")

        raise


def _setup_instance(connection, fsx_dns_name, mount_name):
    """
    Setup FSx mount and VLLM environment on an instance synchronously
    """
    os.chdir("..")
    # Copy script to instance
    connection.put("vllm/ec2/utils/setup_fsx_vllm.sh", "/home/ec2-user/setup_fsx_vllm.sh")

    # Make script executable and run it
    commands = [
        "chmod +x /home/ec2-user/setup_fsx_vllm.sh",
        f"/home/ec2-user/setup_fsx_vllm.sh {fsx_dns_name} {mount_name}",
    ]

    # Execute commands synchronously
    result = connection.run("; ".join(commands))
    return result


def cleanup_resources(ec2_cli, resources, fsx):
    """Cleanup all resources in reverse order of creation"""
    cleanup_errors = []

    def wait_for_instances(instance_ids):
        waiter = ec2_cli.get_waiter("instance_terminated")
        try:
            waiter.wait(InstanceIds=instance_ids, WaiterConfig={"Delay": 60, "MaxAttempts": 100})
            return True
        except WaiterError as e:
            print(f"Warning: Instance termination waiter timed out: {str(e)}")
            return False

    if resources.get("elastic_ips"):
        try:
            ec2_utils.delete_elastic_ips(resources["elastic_ips"], ec2_cli)
            print(f"Deleted elastic IPs: {resources['elastic_ips']}")
        except Exception as e:
            cleanup_errors.append(f"Failed to cleanup Elastic IPs: {str(e)}")

    if resources.get("instances_info"):
        try:
            instance_ids = [instance_id for instance_id, _ in resources["instances_info"]]
            ec2_cli.terminate_instances(InstanceIds=instance_ids)
            print(f"Terminating instances: {instance_ids}")

            if not wait_for_instances(instance_ids):
                cleanup_errors.append("Instances did not terminate within expected timeframe")

            for _, key_filename in resources["instances_info"]:
                if key_filename:
                    try:
                        ec2_cli.delete_key_pair(KeyName=key_filename)
                        for ext in ["", ".pub"]:
                            file_path = f"{key_filename}{ext}"
                            if os.path.exists(file_path):
                                os.remove(file_path)
                    except Exception as e:
                        cleanup_errors.append(f"Failed to delete key file: {str(e)}")
        except Exception as e:
            cleanup_errors.append(f"Failed to cleanup EC2 resources: {str(e)}")

    if resources.get("fsx_config"):
        try:
            fsx.delete_fsx_filesystem(resources["fsx_config"]["filesystem_id"])
            print(f"Deleted FSx filesystem: {resources['fsx_config']['filesystem_id']}")
        except Exception as e:
            cleanup_errors.append(f"Failed to delete FSx filesystem: {str(e)}")

    time.sleep(30)

    if resources.get("sg_fsx"):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                ec2_cli.delete_security_group(GroupId=resources["sg_fsx"])
                print(f"Deleted security group: {resources['sg_fsx']}")
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    cleanup_errors.append(
                        f"Failed to delete security group after {max_attempts} attempts: {str(e)}"
                    )
                else:
                    print(f"Retry {attempt + 1}/{max_attempts} to delete security group")
                    time.sleep(30)

    if cleanup_errors:
        raise Exception("Cleanup errors occurred:\n" + "\n".join(cleanup_errors))


def launch_ec2_instances(ec2_cli, image):
    """Launch EC2 instances with EFA support"""
    instance_type = ec2_instance_type(image)
    ami_id = ec2_instance_ami(DEFAULT_REGION, image)
    az_options = availability_zone_options(ec2_cli, instance_type, DEFAULT_REGION)
    is_arm64 = True if "arm64" in image else False

    instances_info = efa_ec2_instances(
        ec2_client=ec2_cli,
        ec2_instance_type=instance_type,
        ec2_instance_role_name=EC2_INSTANCE_ROLE_NAME,
        ec2_key_name="vllm-ec2-test",
        ec2_instance_ami=ami_id,
        region=DEFAULT_REGION,
        availability_zone_options=az_options,
        is_arm64=is_arm64,
    )
    print(f"Launched instances: {instances_info}")
    return instances_info


def configure_security_groups(instance_id, ec2_cli, fsx, vpc_id, instances_info):
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
        fsx_name = f"fsx-lustre-vllm-ec2-test-sg-{instance_id}-{TEST_ID}"
        # Create FSx security group
        sg_fsx = fsx.create_fsx_security_group(
            ec2_cli,
            vpc_id,
            fsx_name,
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


def mount_fsx_on_worker(instance_id, key_filename, ec2_cli, fsx_dns_name, mount_name):
    """Mount FSx on worker instance without running setup script"""
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

    commands = [
        "sudo yum install -y lustre-client",
        "sudo mkdir -p /fsx",
        f"sudo mount -t lustre -o relatime,flock {fsx_dns_name}@tcp:/{mount_name} /fsx",
    ]

    for cmd in commands:
        connection.run(cmd)


def setup(image):
    """Main setup function for VLLM on EC2 with FSx"""
    print("Testing vllm on ec2........")
    fsx = FsxSetup(DEFAULT_REGION)
    ec2_cli = ec2_client(DEFAULT_REGION)
    resources = {"instances_info": None, "fsx_config": None, "sg_fsx": None}

    try:
        vpc_id = get_default_vpc_id(ec2_cli)
        subnet_ids = get_subnet_id_by_vpc(ec2_cli, vpc_id)

        instance_result = launch_ec2_instances(ec2_cli, image)
        resources["instances_info"] = instance_result["instances"]
        resources["elastic_ips"] = instance_result["elastic_ips"]
        resources["connections"] = instance_result["connections"]
        print("Waiting 60 seconds for instances to initialize...")
        time.sleep(60)

        instance_ids = [instance_id for instance_id, _ in resources["instances_info"]]
        resources["sg_fsx"] = configure_security_groups(
            instance_ids[0], ec2_cli, fsx, vpc_id, resources["instances_info"]
        )

        # Create FSx filesystem
        resources["fsx_config"] = fsx.create_fsx_filesystem(
            subnet_ids[0],
            [resources["sg_fsx"]],
            1200,
            "SCRATCH_2",
            {"Name": f"fsx-lustre-vllm-ec2-test-{instance_ids[0]}-{TEST_ID}"},
        )
        print("Created FSx filesystem")

        master_instance_id, master_key_filename = resources["instances_info"][0]
        setup_instance(
            master_instance_id,
            master_key_filename,
            ec2_cli,
            resources["fsx_config"]["dns_name"],
            resources["fsx_config"]["mount_name"],
        )
        print(f"Setup completed for master instance {master_instance_id}")

        if len(resources["instances_info"]) > 1:
            worker_instance_id, worker_key_filename = resources["instances_info"][1]
            mount_fsx_on_worker(
                worker_instance_id,
                worker_key_filename,
                ec2_cli,
                resources["fsx_config"]["dns_name"],
                resources["fsx_config"]["mount_name"],
            )
            print(f"FSx mounted on worker instance {worker_instance_id}")

        return resources

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        cleanup_resources(ec2_cli, resources, fsx)
        raise


if __name__ == "__main__":
    setup()
