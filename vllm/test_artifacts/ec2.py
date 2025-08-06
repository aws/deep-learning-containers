from test.test_utils.ec2 import get_account_id_from_image_uri, login_to_ecr_registry, get_ec2_client
import time, os, json
from vllm.infra.utils.fsx_utils import FsxSetup
from vllm.infra.ec2 import cleanup_resources
from test.dlc_tests.ec2.test_efa import (
    _setup_multinode_efa_instances,
    EFA_SANITY_TEST_CMD,
    MASTER_CONTAINER_NAME,
    HOSTS_FILE_LOCATION,
    EFA_INTEGRATION_TEST_CMD,
    DEFAULT_EFA_TIMEOUT,
    EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION,
)
from test.test_utils import run_cmd_on_container
from botocore.config import Config
import threading
from fabric import Connection

DEFAULT_REGION = "us-west-2"

import boto3
from botocore.exceptions import ClientError

import time
import logging
from contextlib import contextmanager
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# Helper functions
def setup_env(connection):
    """Setup Python environment on a node"""
    setup_command = """
    python3 -m venv vllm_env && \
    source vllm_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install numpy torch tqdm aiohttp pandas datasets pillow vllm && \
    pip install "transformers[torch]"
    """
    connection.run(setup_command)


def create_head_node_command(image_uri: str, head_ip: str, hf_token: str) -> str:
    """Create command for head node startup"""
    return f"""
    source vllm_env/bin/activate &&
    cd /fsx/vllm-dlc &&
    bash vllm/examples/online_serving/run_cluster.sh \
    {image_uri} {head_ip} \
    --head \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP={head_ip} \
    -e HF_TOKEN={hf_token} \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1 \
    -p 8000:8000
    """


def create_worker_node_command(image_uri: str, head_ip: str, worker_ip: str) -> str:
    """Create command for worker node startup"""
    return f"""
    source vllm_env/bin/activate &&
    cd /fsx/vllm-dlc &&
    bash vllm/examples/online_serving/run_cluster.sh \
    {image_uri} {head_ip} \
    --worker \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP={worker_ip} \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1
    """


def create_serve_command(model_name: str) -> str:
    """Create command for model serving"""
    return f"""
    vllm serve {model_name} \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --max-num-batched-tokens 16384 \
    --port 8000
    """


def create_benchmark_command(model_name: str) -> str:
    """Create command for running benchmark"""
    return f"""
    source vllm_env/bin/activate &&
    python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model {model_name} \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000
    """


def get_secret_hf_token():

    secret_name = "test/hf_token"
    region_name = "us-west-2"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    response = json.loads(get_secret_value_response["SecretString"])

    return response


@contextmanager
def docker_cleanup(connection):
    """Context manager to ensure Docker cleanup"""
    try:
        yield
    finally:
        logger.info("Cleaning up Docker containers...")
        connection.run("docker rm -f $(docker ps -aq)", warn=True)


def wait_for_container_ready(connection, container_id: str, timeout: int = 300) -> bool:
    """
    Wait for container to be ready by checking logs
    Returns True if container is ready, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        logs = connection.run(f"docker logs {container_id}", hide=True).stdout
        if "Ray runtime started" in logs:
            return True
        time.sleep(10)
    return False


def get_container_id(connection, image_name: str) -> Optional[str]:
    """Get container ID for running container with specified image"""
    result = connection.run(f"docker ps -q --filter ancestor={image_name}", hide=True)
    return result.stdout.strip() if result.stdout else None


def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
    """
    Run VLLM benchmark test on multiple EC2 instances using distributed setup

    Args:
        head_connection: Fabric connection to head node
        worker_connection: Fabric connection to worker node
        image_uri: Docker image URI for VLLM container

    Returns:
        dict: Benchmark results

    Raises:
        VLLMBenchmarkError: If benchmark fails
    """
    with docker_cleanup(head_connection), docker_cleanup(worker_connection):
        try:
            # Get HF token and setup configuration
            response = get_secret_hf_token()
            hf_token = response.get("HF_TOKEN")
            if not hf_token:
                raise Exception("Failed to get HF token")

            account_id = get_account_id_from_image_uri(image_uri)
            login_to_ecr_registry(head_connection, account_id, DEFAULT_REGION)
            login_to_ecr_registry(worker_connection, account_id, DEFAULT_REGION)

            print(f"Pulling image: {image_uri}")
            head_connection.run(f"docker pull {image_uri}", hide="out")
            worker_connection.run(f"docker pull {image_uri}", hide="out")

            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            head_ip = head_connection.run("hostname -i").stdout.strip()
            worker_ip = worker_connection.run("hostname -i").stdout.strip()

            # Setup Python environment
            setup_env(head_connection)
            setup_env(worker_connection)

            # Start head node
            logger.info("Starting head node...")
            head_cmd = create_head_node_command(image_uri, head_ip, hf_token)
            head_connection.run(head_cmd, hide=False, asynchronous=True)

            # Wait for head node to be ready
            head_container_id = get_container_id(head_connection, image_uri)
            if not head_container_id or not wait_for_container_ready(
                head_connection, head_container_id
            ):
                raise Exception("Head node failed to start")

            # Start worker node
            logger.info("Starting worker node...")
            worker_cmd = create_worker_node_command(image_uri, head_ip, worker_ip)
            worker_connection.run(worker_cmd, hide=False, asynchronous=True)

            worker_container_id = get_container_id(worker_connection, image_uri)
            if not worker_container_id or not wait_for_container_ready(
                worker_connection, worker_container_id
            ):
                raise Exception("Worker node failed to start")

            # Start model serving
            print("Starting model serving inside Ray container...")
            serve_cmd = create_serve_command(model_name)
            head_container_id = get_container_id(head_connection, image_uri)
            if not head_container_id:
                raise Exception("Cannot find head node container")

            serve_in_container = f"docker exec -it {head_container_id} bash -c '{serve_cmd}'"
            head_connection.run(serve_in_container, hide=False, asynchronous=True)

            print("Waiting for model to load (15 minutes)...")
            time.sleep(900)  # 15 minutes

            # Run benchmark
            logger.info("Running benchmark...")
            benchmark_cmd = create_benchmark_command(model_name)
            result = head_connection.run(benchmark_cmd, timeout=7200)

            return result

        except Exception as e:
            raise Exception(f"Multi-node test execution failed: {str(e)}")


def test_vllm_benchmark_on_single_node(connection, image_uri):
    """
    Run VLLM benchmark test on a single node EC2 instance using the shell script

    Args:
        connection: Fabric connection object to EC2 instance
        image_uri: ECR image URI for VLLM container

    Returns:
        ec2_res: Result object from test execution
    """
    try:
        # Get HF token
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

        account_id = get_account_id_from_image_uri(image_uri)
        login_to_ecr_registry(connection, account_id, DEFAULT_REGION)

        print(f"Pulling image: {image_uri}")
        connection.run(f"docker pull {image_uri}", hide="out")

        # Copy script to instance
        connection.put(
            "vllm/test_artifacts/run_vllm_benchmark_single_node.sh",
            "/home/ec2-user/run_vllm_benchmark_single_node.sh",
        )

        # Make script executable and run it
        commands = [
            "chmod +x /home/ec2-user/run_vllm_benchmark_single_node.sh",
            f"/home/ec2-user/run_vllm_benchmark_single_node.sh {image_uri} {hf_token} {model_name}",
        ]

        # Execute commands synchronously
        result = connection.run(
            "; ".join(commands),
            hide=False,
            timeout=3600,
        )

        return result

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise


def verify_gpu_setup(connection):
    """
    Verify GPU setup on the instance before running the test

    Args:
        connection: Fabric connection object to EC2 instance

    Returns:
        bool: True if GPU setup is valid, False otherwise
    """
    try:
        # Check nvidia-smi
        result = connection.run("nvidia-smi", hide=True)
        if result.failed:
            print("nvidia-smi check failed")
            return False

        # Check CUDA availability
        cuda_check = connection.run("nvidia-smi -L", hide=True)
        if cuda_check.failed or "GPU" not in cuda_check.stdout:
            print("No GPUs found")
            return False

        return True

    except Exception as e:
        print(f"GPU verification failed: {str(e)}")
        return False


def cleanup_containers(connection):
    """
    Cleanup docker containers and images on the instance

    Args:
        connection: Fabric connection object
    """
    try:
        print("Cleaning up containers and images...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")


def run_single_node_test(connection, image_uri):
    """
    Run single node VLLM benchmark test

    Args:
        connection: Fabric connection object
        image_uri: ECR image URI
    """
    try:
        print("\n=== Starting Single-Node Test ===")
        if not verify_gpu_setup(connection):
            raise Exception("GPU setup verification failed")

        result = test_vllm_benchmark_on_single_node(connection, image_uri)
        if result.ok:
            print("Single-node test completed successfully")
            return True
        return False

    finally:
        cleanup_containers(connection)


def run_multi_node_test(head_conn, worker_conn, image_uri):
    """
    Run multi-node VLLM benchmark test

    Args:
        head_conn: Fabric connection object for head node
        worker_conn: Fabric connection object for worker node
        image_uri: ECR image URI
    """
    print("\n=== Starting Multi-Node Test ===")
    try:
        # Verify GPU setup on both nodes
        verification_tasks = [(head_conn, "head"), (worker_conn, "worker")]

        for conn, node_type in verification_tasks:
            if not verify_gpu_setup(conn):
                raise Exception(f"GPU setup verification failed for {node_type} node")

        result = test_vllm_benchmark_on_multi_node(head_conn, worker_conn, image_uri)
        print(result.stdout)
        if result.ok:
            print("Multi-node test completed successfully")
            return True
        return False

    finally:
        for conn in [head_conn, worker_conn]:
            cleanup_containers(conn)


def test_vllm_on_ec2(resources, image_uri):
    """
    Test VLLM on EC2 instances in the following order:
    1. EFA testing
    2. Single node test
    3. Multi-node test

    Args:
        resources: Dictionary containing instance information and FSx config
        image_uri: Docker image URI to test
    """
    ec2_cli = None
    fsx = None
    ec2_connections = {}
    test_results = {"efa": False, "single_node": False, "multi_node": False}

    try:
        ec2_cli = get_ec2_client(DEFAULT_REGION)
        fsx = FsxSetup(DEFAULT_REGION)

        # Create connections
        for instance_id, key_filename in resources["instances_info"]:
            try:
                instance_details = ec2_cli.describe_instances(InstanceIds=[instance_id])[
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
                print(f"Successfully connected to instance {instance_id}")

            except Exception as e:
                print(f"Failed to connect to instance {instance_id}: {str(e)}")
                raise

        if len(ec2_connections) >= 2:
            print("\n=== Starting EFA Tests ===")
            instance_ids = list(ec2_connections.keys())
            number_of_nodes = 2

            _setup_multinode_efa_instances(
                image_uri,
                resources["instances_info"][:2],
                [ec2_connections[instance_ids[0]], ec2_connections[instance_ids[1]]],
                EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION,
                DEFAULT_REGION,
            )

            master_connection = ec2_connections[instance_ids[0]]

            # Run EFA sanity test
            run_cmd_on_container(
                MASTER_CONTAINER_NAME, master_connection, EFA_SANITY_TEST_CMD, hide=False
            )

            run_cmd_on_container(
                MASTER_CONTAINER_NAME,
                master_connection,
                f"{EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} {number_of_nodes}",
                hide=False,
                timeout=DEFAULT_EFA_TIMEOUT,
            )

            test_results["efa"] = True
            for conn in [head_conn, worker_conn]:
                cleanup_containers(conn)
            print("EFA tests completed successfully")

        # Run single-node test on first instance
        # instance_id = list(ec2_connections.keys())[0]
        # print(f"\n=== Running Single-Node Test on instance: {instance_id} ===")
        # test_results["single_node"] = run_single_node_test(ec2_connections[instance_id], image_uri)

        # Run multi-node test if we have at least 2 instances
        if len(ec2_connections) >= 2:
            instance_ids = list(ec2_connections.keys())
            head_conn = ec2_connections[instance_ids[0]]
            worker_conn = ec2_connections[instance_ids[1]]

            test_results["multi_node"] = run_multi_node_test(head_conn, worker_conn, image_uri)
        else:
            print("\nSkipping multi-node test: insufficient instances")

        print("\n=== Test Summary ===")
        print(f"EFA tests: {'Passed' if test_results['efa'] else 'Not Run/Failed'}")
        # print(f"Single-node test: {'Passed' if test_results['single_node'] else 'Failed'}")
        print(f"Multi-node test: {'Passed' if test_results['multi_node'] else 'Failed'}")

        if not any(test_results.values()):
            raise Exception("All tests failed")

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise

    finally:
        if ec2_cli and fsx:
            cleanup_timer = threading.Timer(
                1000, lambda: print("Cleanup timed out, some resources might need manual cleanup")
            )
            cleanup_timer.start()

            try:
                cleanup_resources(
                    ec2_cli,
                    resources["instances_info"],
                    resources["instance_configs"],
                    fsx,
                )
                cleanup_timer.cancel()
                print("Resources cleaned up successfully")
            except Exception as e:
                print(f"Cleanup failed: {str(e)}")
            finally:
                cleanup_timer.cancel()
