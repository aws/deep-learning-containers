"""EC2 EFA instance lifecycle helpers.

Provides a context manager for launching EFA-enabled EC2 instances,
setting up multi-node containers with SSH, and guaranteed cleanup.
"""

import logging
import os
import time
from contextlib import contextmanager

from test_utils import random_suffix_name
from test_utils.aws import AWSSessionManager, LoggedConnection
from test_utils.constants import DEFAULT_REGION, EC2_INSTANCE_ROLE_NAME

LOGGER = logging.getLogger(__name__)

MASTER_CONTAINER_NAME = "master_container"
WORKER_CONTAINER_NAME = "worker_container"
MASTER_SSH_KEY_NAME = "master_id_rsa"
WORKER_SSH_KEY_NAME = "worker_id_rsa"
HOSTS_FILE_LOCATION = "/root/hosts"
DEFAULT_TIMEOUT = 600


def get_efa_devices(conn):
    """Get list of EFA device paths on an instance."""
    result = conn.run("ls -d /dev/infiniband/uverbs* 2>/dev/null || true")
    devices = result.stdout.strip().split()
    return devices


def get_num_gpus(conn):
    """Get number of GPUs on an instance."""
    result = conn.run("nvidia-smi -L | wc -l")
    return int(result.stdout.strip())


def get_num_efa_interfaces(instance_type):
    """Return number of EFA interfaces for known instance types."""
    efa_map = {
        "p4d.24xlarge": 4,
        "p4de.24xlarge": 4,
        "p5.48xlarge": 32,
    }
    return efa_map.get(instance_type, 1)


def generate_efa_network_interfaces(aws_session, instance_type, subnet_id, sg_id):
    """Generate NetworkInterfaces config for EFA-enabled launch."""
    num_interfaces = get_num_efa_interfaces(instance_type)
    interfaces = []
    for idx in range(num_interfaces):
        iface = {
            "DeviceIndex": 0 if idx == 0 else 1,
            "NetworkCardIndex": idx,
            "SubnetId": subnet_id,
            "Groups": [sg_id],
            "InterfaceType": "efa",
            "DeleteOnTermination": True,
        }
        interfaces.append(iface)
    return interfaces


def get_default_subnet(aws_session, az=None):
    """Get a default subnet ID, optionally in a specific AZ."""
    filters = [{"Name": "default-for-az", "Values": ["true"]}]
    if az:
        filters.append({"Name": "availability-zone", "Values": [az]})
    subnets = aws_session.ec2.describe_subnets(Filters=filters)["Subnets"]
    if not subnets:
        raise RuntimeError(f"No default subnet found{f' in {az}' if az else ''}")
    return subnets[0]["SubnetId"]


def create_efa_security_group(aws_session, group_name=None):
    """Create a security group allowing SSH from runner + all traffic within the group (for EFA)."""
    if not group_name:
        group_name = random_suffix_name("dlc-efa", 36)
    vpc_id = aws_session.ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])[
        "Vpcs"
    ][0]["VpcId"]

    response = aws_session.ec2.create_security_group(
        GroupName=group_name,
        Description="Ephemeral EFA test security group",
        VpcId=vpc_id,
    )
    sg_id = response["GroupId"]

    # Allow SSH from CodeBuild runner
    runner_ip = aws_session.get_codebuild_runner_public_ip()
    aws_session.ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": f"{runner_ip}/32"}],
            },
        ],
    )

    # Allow all traffic within the security group (required for EFA + MPI)
    aws_session.ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "-1",
                "UserIdGroupPairs": [{"GroupId": sg_id}],
            },
        ],
    )

    LOGGER.info(f"Created EFA security group {sg_id} ({group_name})")
    return sg_id


def get_availability_zones(aws_session):
    """Get all available AZs in the region."""
    response = aws_session.ec2.describe_availability_zones(
        Filters=[{"Name": "state", "Values": ["available"]}]
    )
    return [az["ZoneName"] for az in response["AvailabilityZones"]]


def get_available_reservations(aws_session, instance_type, min_count=1):
    """Get capacity reservations with available instances, sorted by availability."""
    response = aws_session.ec2.describe_capacity_reservations(
        Filters=[
            {"Name": "instance-type", "Values": [instance_type]},
            {"Name": "state", "Values": ["active"]},
        ]
    )
    reservations = [
        r for r in response["CapacityReservations"] if r["AvailableInstanceCount"] >= min_count
    ]
    reservations.sort(key=lambda r: r["AvailableInstanceCount"])
    return reservations


def _build_efa_run_params(ami_id, instance_type, key_name, network_interfaces, az, name=""):
    """Build common RunInstances params for EFA launch."""
    return {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "NetworkInterfaces": network_interfaces,
        "Placement": {"AvailabilityZone": az},
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
        },
        "BlockDeviceMappings": [
            {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 300}},
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"CI-CD EFA {name}"}],
            },
        ],
        "IamInstanceProfile": {"Name": EC2_INSTANCE_ROLE_NAME},
    }


def launch_efa_instances(aws_session, ami_id, instance_type, key_name, sg_id, count=2, name=""):
    """Launch EFA instances: try capacity reservations first, then on-demand across AZs.

    Returns list of instance IDs.
    """
    from botocore.exceptions import ClientError

    # 1. Try capacity reservations first
    reservations = get_available_reservations(aws_session, instance_type, min_count=count)
    for reservation in reservations:
        az = reservation["AvailabilityZone"]
        cr_id = reservation["CapacityReservationId"]
        subnet_id = get_default_subnet(aws_session, az)
        network_interfaces = generate_efa_network_interfaces(
            aws_session, instance_type, subnet_id, sg_id
        )
        params = _build_efa_run_params(
            ami_id, instance_type, key_name, network_interfaces, az, name
        )
        params["MinCount"] = count
        params["MaxCount"] = count
        params["CapacityReservationSpecification"] = {
            "CapacityReservationTarget": {"CapacityReservationId": cr_id},
        }
        try:
            response = aws_session.ec2.run_instances(**params)
            instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
            LOGGER.info(
                f"Launched {count}x {instance_type} in {az} via reservation {cr_id}: {instance_ids}"
            )
            return instance_ids
        except ClientError as e:
            LOGGER.warning(f"Failed to launch via reservation {cr_id} in {az}: {e}")
            continue

    # 2. Fall back to on-demand, trying each AZ with retries
    LOGGER.info("No capacity reservations available, trying on-demand...")
    max_attempts = 6
    wait_seconds = 300

    for attempt in range(max_attempts):
        azs = get_availability_zones(aws_session)
        for az in azs:
            subnet_id = get_default_subnet(aws_session, az)
            network_interfaces = generate_efa_network_interfaces(
                aws_session, instance_type, subnet_id, sg_id
            )
            params = _build_efa_run_params(
                ami_id, instance_type, key_name, network_interfaces, az, name
            )
            params["MinCount"] = count
            params["MaxCount"] = count
            try:
                response = aws_session.ec2.run_instances(**params)
                instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
                LOGGER.info(f"Launched {count}x {instance_type} in {az} on-demand: {instance_ids}")
                return instance_ids
            except ClientError as e:
                if "InsufficientInstanceCapacity" in str(e) or "Unsupported" in str(e):
                    LOGGER.warning(f"No {instance_type} capacity in {az}, trying next AZ...")
                    continue
                raise

        if attempt < max_attempts - 1:
            LOGGER.warning(
                f"No {instance_type} capacity in any AZ (attempt {attempt + 1}/{max_attempts}). "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"No {instance_type} capacity in any AZ after {max_attempts} attempts")


def setup_container(conn, image_uri, container_name):
    """Pull image and start container with EFA devices and host networking."""
    devices = get_efa_devices(conn)
    device_args = " ".join(f"--device {d}" for d in devices)

    conn.run(f"docker rm -f {container_name}", warn=True)
    conn.run(
        f"docker run --runtime=nvidia --gpus all -id "
        f"--name {container_name} --network host --ulimit memlock=-1:-1 "
        f"{device_args} -v $HOME/test:/test -v /dev/shm:/dev/shm "
        f"{image_uri} bash"
    )
    LOGGER.info(f"Started container {container_name}")


def run_on_container(container_name, conn, cmd, timeout=DEFAULT_TIMEOUT, warn=False):
    """Execute a command inside a running container."""
    return conn.run(f"docker exec {container_name} bash -c '{cmd}'", timeout=timeout, warn=warn)


def setup_master_ssh(conn):
    """Configure SSH client on master container."""
    run_on_container(MASTER_CONTAINER_NAME, conn, f"rm -rf $HOME/.ssh/{MASTER_SSH_KEY_NAME}*")
    run_on_container(
        MASTER_CONTAINER_NAME, conn, f'ssh-keygen -t rsa -f $HOME/.ssh/{MASTER_SSH_KEY_NAME} -N ""'
    )
    ssh_config = (
        "Host *\\n"
        f" IdentityFile /root/.ssh/{MASTER_SSH_KEY_NAME}\\n"
        " StrictHostKeyChecking no\\n"
        " UserKnownHostsFile /dev/null\\n"
        " Port 2022"
    )
    run_on_container(MASTER_CONTAINER_NAME, conn, f'echo -e "{ssh_config}" > $HOME/.ssh/config')
    run_on_container(MASTER_CONTAINER_NAME, conn, "chmod -R 600 $HOME/.ssh/*")


def setup_worker_ssh(conn, master_pub_key):
    """Configure SSH server on worker container to accept master connections."""
    run_on_container(WORKER_CONTAINER_NAME, conn, 'echo "Port 2022" >> /etc/ssh/sshd_config')
    run_on_container(WORKER_CONTAINER_NAME, conn, f"rm -rf $HOME/.ssh/{WORKER_SSH_KEY_NAME}*")
    run_on_container(
        WORKER_CONTAINER_NAME, conn, f'ssh-keygen -t rsa -f $HOME/.ssh/{WORKER_SSH_KEY_NAME} -N ""'
    )
    run_on_container(
        WORKER_CONTAINER_NAME,
        conn,
        f"cp $HOME/.ssh/{WORKER_SSH_KEY_NAME}.pub $HOME/.ssh/authorized_keys",
    )
    run_on_container(
        WORKER_CONTAINER_NAME, conn, f'echo "{master_pub_key}" >> $HOME/.ssh/authorized_keys'
    )
    run_on_container(
        WORKER_CONTAINER_NAME,
        conn,
        f"eval `ssh-agent -s` && ssh-add $HOME/.ssh/{WORKER_SSH_KEY_NAME}",
    )
    run_on_container(WORKER_CONTAINER_NAME, conn, "service ssh start")
    status = run_on_container(WORKER_CONTAINER_NAME, conn, "service ssh status", warn=True)
    if status.failed:
        raise RuntimeError("Failed to start SSH daemon on worker")


def create_hosts_file(master_conn, worker_private_ip, num_gpus):
    """Create MPI hosts file on master container."""
    hosts = f"localhost slots={num_gpus}\n{worker_private_ip} slots={num_gpus}"
    run_on_container(
        MASTER_CONTAINER_NAME, master_conn, f'echo -e "{hosts}" > {HOSTS_FILE_LOCATION}'
    )


def get_private_ip(aws_session, instance_id):
    """Get private IP of an instance."""
    response = aws_session.ec2.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]["PrivateIpAddress"]


def allocate_and_associate_eip(aws_session, instance_id):
    """Allocate an Elastic IP and associate it with the instance's primary network interface.

    Returns (allocation_id, public_ip).
    """
    eip = aws_session.ec2.allocate_address(Domain="vpc")
    alloc_id = eip["AllocationId"]
    public_ip = eip["PublicIp"]

    # Get the primary network interface (DeviceIndex 0)
    instance = aws_session.ec2.describe_instances(InstanceIds=[instance_id])
    eni_id = None
    for iface in instance["Reservations"][0]["Instances"][0]["NetworkInterfaces"]:
        if iface["Attachment"]["DeviceIndex"] == 0:
            eni_id = iface["NetworkInterfaceId"]
            break

    aws_session.ec2.associate_address(
        AllocationId=alloc_id,
        NetworkInterfaceId=eni_id,
    )
    LOGGER.info(f"Associated EIP {public_ip} ({alloc_id}) with instance {instance_id}")
    return alloc_id, public_ip


def release_eip(aws_session, alloc_id):
    """Release an Elastic IP."""
    try:
        aws_session.ec2.release_address(AllocationId=alloc_id)
        LOGGER.info(f"Released EIP {alloc_id}")
    except Exception as e:
        LOGGER.warning(f"Failed to release EIP {alloc_id}: {e}")


@contextmanager
def efa_instances(image_uri, instance_type="p4d.24xlarge", region=DEFAULT_REGION):
    """Context manager that launches 2 EFA instances, sets up containers + SSH, and cleans up.

    Yields (master_conn, worker_conn, aws_session) where connections are to the EC2 hosts.
    """
    aws_session = AWSSessionManager(region=region)
    ami_id = aws_session.get_latest_ami()
    key_name, key_path = aws_session.create_key_pair()
    sg_id = create_efa_security_group(aws_session)

    master_id = None
    worker_id = None
    master_eip_alloc = None
    worker_eip_alloc = None

    try:
        # Launch both instances in the same AZ (tries each AZ on capacity errors)
        instance_ids = launch_efa_instances(
            aws_session, ami_id, instance_type, key_name, sg_id, count=2, name="efa-test"
        )
        master_id = instance_ids[0]
        worker_id = instance_ids[1]

        # Wait for both
        aws_session.wait_for_instance_ready(master_id)
        aws_session.wait_for_instance_ready(worker_id)

        # Allocate Elastic IPs (EFA multi-NIC instances don't get auto public IPs)
        master_eip_alloc, master_ip = allocate_and_associate_eip(aws_session, master_id)
        worker_eip_alloc, worker_ip = allocate_and_associate_eip(aws_session, worker_id)

        # SSH connections using EIP addresses
        master_conn = LoggedConnection(
            host=master_ip,
            user="ec2-user",
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )
        worker_conn = LoggedConnection(
            host=worker_ip,
            user="ec2-user",
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )

        # Copy test scripts to instances
        master_conn.run("mkdir -p ~/test/efa/scripts ~/test/efa/logs")
        worker_conn.run("mkdir -p ~/test/efa/scripts ~/test/efa/logs")
        scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
        for script in os.listdir(scripts_dir):
            master_conn.put(os.path.join(scripts_dir, script), f"~/test/efa/scripts/{script}")
            worker_conn.put(os.path.join(scripts_dir, script), f"~/test/efa/scripts/{script}")
        master_conn.run("chmod +x ~/test/efa/scripts/*.sh")
        worker_conn.run("chmod +x ~/test/efa/scripts/*.sh")

        # ECR login + pull image on both
        account_id = image_uri.split(".")[0]
        ecr_login = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
        master_conn.run(ecr_login)
        worker_conn.run(ecr_login)
        master_conn.run(f"docker pull {image_uri}")
        worker_conn.run(f"docker pull {image_uri}")

        # Start containers with EFA devices
        setup_container(master_conn, image_uri, MASTER_CONTAINER_NAME)
        setup_container(worker_conn, image_uri, WORKER_CONTAINER_NAME)

        # Configure SSH between containers
        setup_master_ssh(master_conn)
        worker_private_ip = get_private_ip(aws_session, worker_id)
        master_pub_key = run_on_container(
            MASTER_CONTAINER_NAME, master_conn, f"cat $HOME/.ssh/{MASTER_SSH_KEY_NAME}.pub"
        ).stdout.strip()
        setup_worker_ssh(worker_conn, master_pub_key)

        # Create MPI hosts file
        num_gpus = get_num_gpus(master_conn)
        create_hosts_file(master_conn, worker_private_ip, num_gpus)

        yield master_conn, worker_conn, aws_session

    finally:
        # Guaranteed cleanup
        if master_id:
            aws_session.terminate_instance(master_id)
        if worker_id:
            aws_session.terminate_instance(worker_id)
        # Release Elastic IPs
        if master_eip_alloc:
            release_eip(aws_session, master_eip_alloc)
        if worker_eip_alloc:
            release_eip(aws_session, worker_eip_alloc)
        # Wait briefly for instances to start terminating before deleting SG
        time.sleep(30)
        try:
            aws_session.delete_security_group(sg_id)
        except Exception as e:
            LOGGER.warning(f"Failed to delete security group {sg_id}: {e}")
        aws_session.delete_key_pair(key_name, key_path)
