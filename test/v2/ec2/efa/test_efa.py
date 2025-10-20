import os

import pytest

import test.test_utils.ec2 as ec2_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX_V2,
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    is_pr_context,
    is_efa_dedicated,
    are_heavy_instance_ec2_tests_enabled,
    login_to_ecr_registry,
    run_cmd_on_container,
)
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from infra.test_infra.ec2.utils import (
    get_efa_ec2_instance_type,
    filter_efa_instance_type,
    filter_efa_only_p4_instance_type,
)
from infra.test_infra.test_infra_utils import create_logger

LOGGER = create_logger(__name__)

BUILD_ALL_REDUCE_PERF_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX_V2, "efa", "build_all_reduce_perf.sh"
)
EFA_SANITY_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX_V2, "efa", "testEFASanity")
EFA_INTEGRATION_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX_V2, "efa", "testEFA")
EFA_PYTORCH_HEALTHCHECK_TEST_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX_V2, "healthcheck_tests", "efa_checker_single_node.sh"
)

ENABLE_IPV6_TESTING = os.getenv("ENABLE_IPV6_TESTING", "false").lower() == "true"

MASTER_SSH_KEY_NAME = "master_id_rsa"
WORKER_SSH_KEY_NAME = "worker_id_rsa"
MASTER_CONTAINER_NAME = "master_container"
WORKER_CONTAINER_NAME = "worker_container"
HOSTS_FILE_LOCATION = "/root/hosts"

DEFAULT_EFA_TIMEOUT = 300


def get_efa_container_name(framework, test_scenario, arch_type, node_role=None):
    """
    Generate unique container name for EC2 EFA tests

    Args:
        framework: Framework name (e.g., "vllm", "pytorch", "tensorflow")
        test_scenario: Test scenario - "efa"
        arch_type: Architecture from buildspec (e.g., "x86_64", "arm64")
        node_role: For multi-node: "master", "worker-0", etc.

    Returns:
        Container name like "vllm-ec2-efa-x86_64-master"
    """
    # Try to get framework from environment variable first
    detected_framework = os.environ.get("FRAMEWORK")
    if not detected_framework:
        detected_framework = framework

    base_name = f"{detected_framework}-ec2-{test_scenario}-{arch_type}"
    return f"{base_name}-{node_role}" if node_role else base_name


EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_instance_type,
)

EC2_EFA_GPU_ONLY_P4_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=filter_efa_only_p4_instance_type,
)


# TODO: decide on whether to keep this commented out or left out until actual implementation of each framework
# def test_pytorch_efa(
#     pytorch_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region, gpu_only
# ):
#     """
#     Run EFA Sanity tests on DLC, and then run NCCL Message Transfer and All Reduce tests using EFA
#     on multiple nodes using DLC images. The test scripts are agnostic to the framework and version
#     installed in the DLC image. The test also builds nccl-tests to create the all_reduce_perf
#     binary necessary for multinode tests, on each node.
#     Note: This test must be explicitly enabled on CI, and will only run on EFA-capable instances
#     on pipelines.
#     :param pytorch_training: str PyTorch Training DLC image URI
#     :param efa_ec2_instances: list of tuples of instance-ids and SSH-keys for EFA-enabled instances
#     :param efa_ec2_connections: list of Fabric Connection objects for EFA-enabled instances
#     :param ec2_instance_type: str Instance Type being tested
#     :param region: str Region in which EFA-enabled instances are launched
#     :param gpu_only: pytest fixture to limit test only to GPU DLCs
#     """
#     number_of_nodes = 2
#     _setup_multinode_efa_instances(
#         pytorch_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region
#     )
#     master_connection = efa_ec2_connections[0]
#     run_cmd_on_container(MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD, hide=False)

#     ipv6_arg = "True" if ENABLE_IPV6_TESTING else ""

#     run_cmd_on_container(
#         MASTER_CONTAINER_NAME,
#         master_connection,
#         f"{EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} {number_of_nodes} {ipv6_arg}",
#         hide=False,
#         timeout=DEFAULT_EFA_TIMEOUT,
#     )


# def test_efa_tensorflow(
#     tensorflow_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region, gpu_only
# ):
#     """
#     Run EFA Sanity tests on DLC, and then run NCCL Message Transfer and All Reduce tests using EFA
#     on multiple nodes using DLC images. The test scripts are agnostic to the framework and version
#     installed in the DLC image. The test also builds nccl-tests to create the all_reduce_perf
#     binary necessary for multinode tests, on each node.
#     Note: This test must be explicitly enabled on CI, and will only run on EFA-capable instances
#     on pipelines.
#     :param tensorflow_training: str PyTorch Training DLC image URI
#     :param efa_ec2_instances: list of tuples of instance-ids and SSH-keys for EFA-enabled instances
#     :param efa_ec2_connections: list of Fabric Connection objects for EFA-enabled instances
#     :param ec2_instance_type: str Instance Type being tested
#     :param region: str Region in which EFA-enabled instances are launched
#     :param gpu_only: pytest fixture to limit test only to GPU DLCs
#     """
#     number_of_nodes = 2
#     _setup_multinode_efa_instances(
#         tensorflow_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region
#     )
#     master_connection = efa_ec2_connections[0]
#     run_cmd_on_container(MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD, hide=False)

#     # pass IPv6 flag if enabled
#     ipv6_arg = "True" if ENABLE_IPV6_TESTING else ""

#     run_cmd_on_container(
#         MASTER_CONTAINER_NAME,
#         master_connection,
#         f"export CUDA_HOME='/usr/local/cuda'; {EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} {number_of_nodes} {ipv6_arg}",
#         hide=False,
#         timeout=DEFAULT_EFA_TIMEOUT,
#     )


# def test_pytorch_efa_healthcheck(
#     pytorch_training,
#     efa_ec2_instances,
#     efa_ec2_connections,
#     ec2_instance_type,
#     region,
#     gpu_only,
# ):
#     """
#     Run EFA Health Check tests on DLC.
#     :param pytorch_training: str PyTorch Training DLC image URI
#     :param efa_ec2_instances: list of tuples of instance-ids and SSH-keys for EFA-enabled instances
#     :param efa_ec2_connections: list of Fabric Connection objects for EFA-enabled instances
#     :param ec2_instance_type: str Instance Type being tested
#     :param region: str Region in which EFA-enabled instances are launched
#     :param gpu_only: pytest fixture to limit test only to GPU DLCs
#     """
#     _setup_multinode_efa_instances(
#         pytorch_training, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region
#     )
#     master_connection = efa_ec2_connections[0]
#     run_cmd_on_container(MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD, hide=False)
#     run_cmd_on_container(
#         MASTER_CONTAINER_NAME,
#         master_connection,
#         f"{EFA_PYTORCH_HEALTHCHECK_TEST_CMD}",
#         hide=False,
#         timeout=DEFAULT_EFA_TIMEOUT,
#     )


def _setup_multinode_efa_instances(
    image, efa_ec2_instances, efa_ec2_connections, ec2_instance_type, region, arch_type=None
):
    """
    Pull and start test image containers on both master and worker instances, configure
    password-less SSH between master and worker nodes, and build all_reduce_perf binary on
    master and worker nodes.
    :param image: str DLC image URI to be tested
    :param efa_ec2_instances: list of tuples of instance_id, keypair_filepath for each instance
    :param efa_ec2_connections: list of fabric connection objects
    :param ec2_instance_type: str instance type being used
    :param region: str region name in which test is being run
    :param arch_type: str architecture type (e.g., "x86_64", "arm64")
    """
    # Asynchronously pull docker image on all instances
    _pull_image_on_all_instances(efa_ec2_connections, image)
    # Configure master node container
    master_connection = efa_ec2_connections[0]

    # Determine container names - use unique names for vLLM, standard names for others
    if "vllm" in image:
        # Use provided arch_type or infer from image as fallback
        if arch_type is None:
            arch_type = "arm64" if "arm64" in image else "x86_64"
        master_container_name = get_efa_container_name("vllm", "efa", arch_type, "master")
    else:
        master_container_name = MASTER_CONTAINER_NAME

    LOGGER.info(f"Master container name: {master_container_name}")

    build_all_reduce_perf_promises = []
    # Run container
    _setup_container(master_connection, image, master_container_name)
    
    # Uncomment to verify container file structure in case of path issues
    # LOGGER.info(f"Verifying files inside {master_container_name} container")
    # run_cmd_on_container(
    #     master_container_name,
    #     master_connection,
    #     "ls -la /test/v2/ec2/efa/",
    #     hide=False,
    # )

    # Build all_reduce_perf binary using nccl-tests
    promise = run_cmd_on_container(
        master_container_name,
        master_connection,
        BUILD_ALL_REDUCE_PERF_CMD,
        timeout=DEFAULT_EFA_TIMEOUT,
        asynchronous=True,
    )
    build_all_reduce_perf_promises.append(promise)

    for idx, worker_connection in enumerate(efa_ec2_connections[1:]):
        # Determine worker container name
        if "vllm" in image:
            worker_container_name = get_efa_container_name(
                "vllm", "efa", arch_type, f"worker-{idx}"
            )
        else:
            worker_container_name = WORKER_CONTAINER_NAME

        LOGGER.info(f"Worker container name: {worker_container_name}")

        # Run container
        _setup_container(worker_connection, image, worker_container_name)
        # Build all_reduce_perf binary using nccl-tests
        promise = run_cmd_on_container(
            worker_container_name,
            worker_connection,
            BUILD_ALL_REDUCE_PERF_CMD,
            timeout=DEFAULT_EFA_TIMEOUT,
            asynchronous=True,
        )
        build_all_reduce_perf_promises.append(promise)

    # Configure master node SSH client-side configurations
    _setup_master_efa_ssh_config(master_connection, master_container_name)
    # Create a hosts file that provides mpi with IP addresses and no. of GPUs in each node
    worker_instance_ids = [instance_id for instance_id, _ in efa_ec2_instances[1:]]
    _create_master_mpi_hosts_file(
        efa_ec2_connections, worker_instance_ids, ec2_instance_type, region, master_container_name
    )
    # Obtain master node SSH public key for future use
    master_pub_key = run_cmd_on_container(
        master_container_name, master_connection, f"cat $HOME/.ssh/{MASTER_SSH_KEY_NAME}.pub"
    ).stdout.strip("\n")

    # Configure worker node containers
    for idx, worker_connection in enumerate(efa_ec2_connections[1:]):
        # Determine worker container name
        if "vllm" in image:
            worker_container_name = get_efa_container_name(
                "vllm", "efa", arch_type, f"worker-{idx}"
            )
        else:
            worker_container_name = WORKER_CONTAINER_NAME

        # Configure worker node SSH server-side configurations, launch SSH daemon, and allow
        # password-less SSH access from master to worker nodes.
        _setup_worker_efa_ssh_config(worker_connection, master_pub_key, worker_container_name)

    # Wait for all_reduce_perf binaries to be built in all containers
    for promise in build_all_reduce_perf_promises:
        promise.join()


def _pull_image_on_all_instances(connections, docker_image):
    """
    Asynchronously pull tested image on all instances
    :param connections: list of Fabric Connection objects
    :param docker_image: str DLC image URI to be tested
    """
    account_id = get_account_id_from_image_uri(docker_image)
    region = get_region_from_image_uri(docker_image)

    for conn in connections:
        login_to_ecr_registry(conn, account_id, region)

    promises = [conn.run(f"docker pull {docker_image}", asynchronous=True) for conn in connections]
    for prom in promises:
        prom.join()


def _setup_container(connection, docker_image, container_name):
    """
    Pull and run tested image with all EFA devices made available to container
    :param connection: Fabric Connection object
    :param docker_image: str DLC image URI to be tested
    :param container_name: str container name
    """
    devices = ec2_utils.get_efa_devices_on_instance(connection)
    docker_devices_args = [f"--device {device_location}" for device_location in devices]
    docker_all_devices_arg = " ".join(docker_devices_args)

    # Remove pre-existing containers if reusing an instance
    connection.run(f"docker rm -f {container_name}", hide=True)

    # Use network mode host, rather than the default "bridge" to allow direct access to container
    # using SSH on a pre-defined port (as decided by sshd_config on server-side).
    # Allow instance to share all memory with container using memlock=-1:-1.
    # Share all EFA devices with container using --device <device_location> for all EFA devices.
    if "vllm" in docker_image:
        connection.run(
            f"docker run --entrypoint=/bin/bash -e CUDA_HOME=/usr/local/cuda --runtime=nvidia --gpus all -id --name {container_name} --network host --ulimit memlock=-1:-1 "
            f"{docker_all_devices_arg} -v $HOME/test/v2:/test/v2 -v /dev/shm:/dev/shm {docker_image}"
        )
    else:
        connection.run(
            f"docker run --runtime=nvidia --gpus all -id --name {container_name} --network host --ulimit memlock=-1:-1 "
            f"{docker_all_devices_arg} -v $HOME/test/v2:/test/v2 -v /dev/shm:/dev/shm {docker_image} bash"
        )
    
    LOGGER.info(f"Container {container_name} started successfully")


def _setup_master_efa_ssh_config(connection, master_container_name):
    """
    Set up SSH client config on master container to connect to worker
    :param connection: Fabric Connection object
    :param master_container_name: str master container name
    """
    run_cmd_on_container(
        master_container_name, connection, f"rm -rf $HOME/.ssh/{MASTER_SSH_KEY_NAME}*"
    )
    # When running container in --network=host, the container hostname changes, requiring
    # a new SSH key to be generated.
    run_cmd_on_container(
        master_container_name,
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
        master_container_name,
        connection,
        f"""echo -e "{master_container_ssh_config}" > $HOME/.ssh/config""",
    )
    run_cmd_on_container(master_container_name, connection, "chmod -R 600 $HOME/.ssh/*")


def _create_master_mpi_hosts_file(
    efa_ec2_connections, worker_instance_ids, instance_type, region, master_container_name
):
    """
    Create MPI Hosts file that contains private IP addresses of all hosts used in training job.
    :param efa_ec2_connections: List of Fabric Connection objects [master_connection, *worker_connections]
    :param worker_instance_ids: list of str worker instance IDs
    :param instance_type: str EC2 Instance Type being used
    :param region: str region name in which test is run
    :param master_container_name: str master container name
    """
    master_connection = efa_ec2_connections[0]
    slots = ec2_utils.get_instance_num_gpus(instance_type=instance_type)
    worker_instance_private_ips = [
        ec2_utils.get_private_ip(instance_id, region) for instance_id in worker_instance_ids
    ]

    if ENABLE_IPV6_TESTING:
        master_ip = master_connection.ipv6_address
        if not master_ip:
            raise RuntimeError("IPv6 testing enabled but no IPv6 address found for master node")

        worker_ips = [conn.ipv6_address for conn in efa_ec2_connections[1:]]
        if not all(worker_ips):
            raise RuntimeError("IPv6 testing enabled but not all workers have IPv6 addresses")

        hosts_string = f"compute1 slots={slots} "
        etc_string = f"{master_ip} compute1"
        compute_counter = 2

        for worker_ip in worker_ips:
            compute_name = f"compute{compute_counter}"
            hosts_string += f"\n{compute_name} slots={slots} "
            etc_string += f"\n{worker_ip} {compute_name}"
            compute_counter += 1

        run_cmd_on_container(
            master_container_name, master_connection, f"""echo "{etc_string}" > /etc/hosts"""
        )

        run_cmd_on_container(
            master_container_name,
            master_connection,
            f"""echo -e "{hosts_string}" > {HOSTS_FILE_LOCATION}""",
        )
    else:
        # Configure MPI hosts file with IP addresses and slots for worker nodes
        hosts_string = f"localhost slots={slots} "
        for worker_ip in worker_instance_private_ips:
            hosts_string += f"\n{worker_ip} slots={slots} "

        run_cmd_on_container(
            master_container_name,
            master_connection,
            f"""echo -e "{hosts_string}" > {HOSTS_FILE_LOCATION}""",
        )


def _setup_worker_efa_ssh_config(connection, master_pub_key, worker_container_name):
    """
    Set up SSH server config on worker container to allow connections from master.
    :param connection: Fabric Connection object
    :param master_pub_key: str Master node public SSH key to allow password-less SSH access
    :param worker_container_name: str worker container name
    """
    # Force SSH Daemon to use port 2022, since port 22 is already in use by the host instance
    run_cmd_on_container(
        worker_container_name, connection, """echo "Port 2022" >> /etc/ssh/sshd_config"""
    )
    run_cmd_on_container(
        worker_container_name, connection, f"rm -rf $HOME/.ssh/{WORKER_SSH_KEY_NAME}*"
    )
    # When running container in --network=host, the container hostname changes, requiring
    # a new SSH key to be generated.
    run_cmd_on_container(
        worker_container_name,
        connection,
        f"""ssh-keygen -t rsa -f $HOME/.ssh/{WORKER_SSH_KEY_NAME} -N "" """,
    )
    # Add both self and master public keys to authorized keys to allow password-less access to
    # this container from authorized hosts.
    run_cmd_on_container(
        worker_container_name,
        connection,
        f"cp $HOME/.ssh/{WORKER_SSH_KEY_NAME}.pub $HOME/.ssh/authorized_keys",
    )
    run_cmd_on_container(
        worker_container_name,
        connection,
        f"""echo "{master_pub_key}" >> $HOME/.ssh/authorized_keys""",
    )
    # Check if ssh agent is running or not, and if not, run it.
    run_cmd_on_container(
        worker_container_name,
        connection,
        f"eval `ssh-agent -s` && ssh-add $HOME/.ssh/{WORKER_SSH_KEY_NAME}",
    )
    # Start SSH service which uses configurations from /etc/ssh/sshd_config
    run_cmd_on_container(worker_container_name, connection, "service ssh start")
    # Check status of SSH service, and fail test-setup if service doesn't run correctly.
    ssh_status = run_cmd_on_container(
        worker_container_name, connection, "service ssh status", warn=True
    )
    if ssh_status.failed:
        raise RuntimeError("Failed to setup SSH Daemon on worker node")
