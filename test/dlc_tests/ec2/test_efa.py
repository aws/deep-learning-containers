import os

import pytest

import test.test_utils.ec2 as ec2_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    login_to_ecr_registry,
    run_cmd_on_container,
)

EFA_SANITY_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "efa", "testEFASanity")
EFA_INTEGRATION_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "efa", "testEFA")

MASTER_SSH_KEY_NAME = "master_id_rsa"
WORKER_SSH_KEY_NAME = "worker_id_rsa"
MASTER_CONTAINER_NAME = "master_container"
WORKER_CONTAINER_NAME = "worker_container"
HOSTS_FILE_LOCATION = "/root/hosts"


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.integration("efa")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.multinode(2)
@pytest.mark.parametrize("region", ["us-west-2"])
@pytest.mark.parametrize("ec2_instance_type", ["p4d.24xlarge"])
def test_efa(pytorch_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, gpu_only):
    _setup_multinode_efa_instances(
        pytorch_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type
    )
    master_connection = efa_ec2_connections[0]
    run_cmd_on_container(MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD)
    run_cmd_on_container(
        MASTER_CONTAINER_NAME,
        master_connection,
        f"{EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} /opt/amazon/openmpi/bin/mpirun"
    )


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.integration("efa")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.multinode(2)
@pytest.mark.parametrize("region", ["us-west-2"])
@pytest.mark.parametrize("ec2_instance_type", ["p4d.24xlarge"])
def test_efa_tensorflow(
    tensorflow_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, gpu_only
):
    _setup_multinode_efa_instances(
        tensorflow_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type
    )
    master_connection = efa_ec2_connections[0]
    run_cmd_on_container(MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD)
    run_cmd_on_container(
        MASTER_CONTAINER_NAME,
        master_connection,
        f"{EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} /opt/amazon/openmpi/bin/mpirun"
    )


def _setup_multinode_efa_instances(
    image, efa_ec2_instances, efa_ec2_connections, ec2_instance_type
):
    # Configure master node container
    master_connection = efa_ec2_connections[0]
    # Run docker login, and pull and run container
    _setup_container(master_connection, image, MASTER_CONTAINER_NAME)
    # Configure master node SSH client-side configurations
    _setup_master_efa_ssh_config(master_connection)
    # Create a hosts file that provides mpi with IP addresses and no. of GPUs in each node
    worker_instance_ids = [instance_id for instance_id, _ in efa_ec2_instances[1:]]
    _create_master_mpi_hosts_file(master_connection, worker_instance_ids, ec2_instance_type)
    # Obtain master node SSH public key for future use
    master_pub_key = run_cmd_on_container(
        MASTER_CONTAINER_NAME, master_connection, f"cat $HOME/.ssh/{MASTER_SSH_KEY_NAME}.pub"
    ).stdout.strip("\n")

    # Configure worker node containers
    for worker_connection in efa_ec2_connections[1:]:
        # Run docker login, and pull and run container
        _setup_container(worker_connection, image, WORKER_CONTAINER_NAME)
        # Configure worker node SSH server-side configurations, launch SSH daemon, and allow
        # password-less SSH access from master to worker nodes.
        _setup_worker_efa_ssh_config(worker_connection, master_pub_key)


def _setup_container(connection, docker_image, container_name):
    account_id = get_account_id_from_image_uri(docker_image)
    region = get_region_from_image_uri(docker_image)
    login_to_ecr_registry(connection, account_id, region)
    connection.run(f"docker pull {docker_image}", hide="out")

    devices = ec2_utils.get_efa_devices_on_instance(connection)
    docker_devices_args = [f"--device {device_location}" for device_location in devices]
    docker_all_devices_arg = " ".join(docker_devices_args)

    # Remove pre-existing containers if reusing an instance
    connection.run(f"docker rm -f {container_name}")

    # Run docker container with nvidia-docker to give access to all GPUs
    # Use network mode host, rather than the default "bridge" to allow direct access to container
    # using SSH on a pre-defined port (as decided by sshd_config on server-side).
    # Allow instance to share all memory with container using memlock=-1:-1.
    # Share all EFA devices with container using --device <device_location> for all EFA devices.
    connection.run(
        f"nvidia-docker run -id --name {container_name} --network host --ulimit memlock=-1:-1 "
        f"{docker_all_devices_arg} -v $HOME/container_tests:/test {docker_image} bash"
    )


def _setup_master_efa_ssh_config(connection):
    run_cmd_on_container(
        MASTER_CONTAINER_NAME, connection, f"rm -rf $HOME/.ssh/{MASTER_SSH_KEY_NAME}*"
    )
    # When running container in --network=host, the container hostname changes, requiring
    # a new SSH key to be generated.
    run_cmd_on_container(
        MASTER_CONTAINER_NAME,
        connection,
        f"""ssh-keygen -t rsa -f $HOME/.ssh/{MASTER_SSH_KEY_NAME} -N "" """,
    )
    # Configure SSH client-side to always use newly created key, and use port 2022, since this is
    # the port configured in the worker node SSH daemon.
    master_container_ssh_config = (
        "Host *\n"
        f" IdentityFile /root/.ssh/{MASTER_SSH_KEY_NAME}\n"
        " StrictHostKeyChecking no\n"
        " UserKnownHostsFile /dev/null\n"
        " Port 2022"
    )
    run_cmd_on_container(
        MASTER_CONTAINER_NAME,
        connection,
        f"""echo -e "{master_container_ssh_config}" > $HOME/.ssh/config""",
    )
    run_cmd_on_container(MASTER_CONTAINER_NAME, connection, "chmod -R 600 $HOME/.ssh/*")


def _create_master_mpi_hosts_file(connection, worker_instance_ids, instance_type):
    slots = ec2_utils.get_instance_num_gpus(instance_type=instance_type)
    worker_instance_private_ips = [
        ec2_utils.get_private_ip(instance_id) for instance_id in worker_instance_ids
    ]
    # Configure MPI hosts file with IP addresses and slots for worker nodes
    hosts_string = f"localhost slots={slots} "
    for worker_ip in worker_instance_private_ips:
        hosts_string += f"\n{worker_ip} slots={slots} "
    run_cmd_on_container(
        MASTER_CONTAINER_NAME, connection, f"""echo -e "{hosts_string}" > {HOSTS_FILE_LOCATION}"""
    )


def _setup_worker_efa_ssh_config(connection, master_pub_key):
    # Force SSH Daemon to use port 2022, since port 22 is already in use by the host instance
    run_cmd_on_container(
        WORKER_CONTAINER_NAME, connection, """echo "Port 2022" >> /etc/ssh/sshd_config"""
    )
    run_cmd_on_container(
        WORKER_CONTAINER_NAME, connection, f"rm -rf $HOME/.ssh/{WORKER_SSH_KEY_NAME}*"
    )
    # When running container in --network=host, the container hostname changes, requiring
    # a new SSH key to be generated.
    run_cmd_on_container(
        WORKER_CONTAINER_NAME,
        connection,
        f"""ssh-keygen -t rsa -f $HOME/.ssh/{WORKER_SSH_KEY_NAME} -N "" """,
    )
    # Add both self and master public keys to authorized keys to allow password-less access to
    # this container from authorized hosts.
    run_cmd_on_container(
        WORKER_CONTAINER_NAME,
        connection,
        f"cp $HOME/.ssh/{WORKER_SSH_KEY_NAME}.pub $HOME/.ssh/authorized_keys",
    )
    run_cmd_on_container(
        WORKER_CONTAINER_NAME,
        connection,
        f"""echo "{master_pub_key}" >> $HOME/.ssh/authorized_keys""",
    )
    # Check if ssh agent is running or not, and if not, run it.
    run_cmd_on_container(
        WORKER_CONTAINER_NAME,
        connection,
        f"eval `ssh-agent -s` && ssh-add $HOME/.ssh/{WORKER_SSH_KEY_NAME}",
    )
    # Start SSH service which uses configurations from /etc/ssh/sshd_config
    run_cmd_on_container(WORKER_CONTAINER_NAME, connection, "service ssh start")
    # Check status of SSH service, and fail test-setup if service doesn't run correctly.
    ssh_status = run_cmd_on_container(
        WORKER_CONTAINER_NAME, connection, "service ssh status", warn=True
    )
    if ssh_status.failed:
        raise RuntimeError("Failed to setup SSH Daemon on worker node")
