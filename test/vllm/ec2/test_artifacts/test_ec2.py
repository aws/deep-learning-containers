import threading
import boto3
import time, os, json
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
from fabric import Connection
from contextlib import contextmanager
from typing import Optional, Tuple

from test.test_utils.ec2 import (
    get_account_id_from_image_uri,
    login_to_ecr_registry,
    get_ec2_client,
)
from test.vllm.ec2.utils.fsx_utils import FsxSetup
from test.vllm.ec2.infra.setup_ec2 import cleanup_resources, TEST_ID
from test.dlc_tests.ec2.test_efa import (
    _setup_multinode_efa_instances,
    EFA_SANITY_TEST_CMD,
    MASTER_CONTAINER_NAME,
    HOSTS_FILE_LOCATION,
    EFA_INTEGRATION_TEST_CMD,
    DEFAULT_EFA_TIMEOUT,
)
from test.test_utils import run_cmd_on_container

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_REGION = "us-west-2"
logger = logging.getLogger(__name__)


def setup_env(connection):
    """Setup Python environment on a node"""
    setup_command = """
    python3 -m venv vllm_env && \
    source vllm_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install numpy torch tqdm aiohttp pandas datasets pillow ray vllm==0.10.0 && \
    pip install "transformers<4.54.0"
    """
    connection.run(setup_command, shell=True)


def create_benchmark_command() -> str:
    """Create command for running benchmark"""
    return f"""
    python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model {MODEL_NAME} \
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


def wait_for_container_ready(connection, timeout: int = 1000) -> bool:
    """
    Wait for container and model to be ready by checking logs and endpoint
    Returns True if container and model are ready, False if timeout
    """
    start_time = time.time()
    model_ready = False

    while time.time() - start_time < timeout:
        if not model_ready:
            try:
                curl_cmd = """
                curl -s http://localhost:8000/v1/completions \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    "prompt": "Hello",
                    "max_tokens": 10
                }'
                """
                result = connection.run(curl_cmd, hide=False)
                if result.ok:
                    print("Model endpoint is responding")
                    model_ready = True
                    return True
            except Exception:
                pass
    return False


def setup_docker_image(conn, image_uri):
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(conn, account_id, DEFAULT_REGION)
    print(f"Pulling image: {image_uri}")
    conn.run(f"docker pull {image_uri}", hide="out")


def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
    try:
        # Get HF token
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        if not hf_token:
            raise Exception("Failed to get HF token")

        for conn in [head_connection, worker_connection]:
            setup_docker_image(conn, image_uri)

        head_connection.put(
            "vllm/ec2/utils/head_node_setup.sh", "/home/ec2-user/head_node_setup.sh"
        )
        worker_connection.put(
            "vllm/ec2/utils/worker_node_setup.sh", "/home/ec2-user/worker_node_setup.sh"
        )

        head_connection.run("chmod +x head_node_setup.sh")
        worker_connection.run("chmod +x worker_node_setup.sh")

        head_ip = head_connection.run("hostname -i").stdout.strip()
        worker_ip = worker_connection.run("hostname -i").stdout.strip()

        container_name = "ray_head-" + TEST_ID
        print("Starting head node...")
        head_connection.run(
            f"./head_node_setup.sh {image_uri} {hf_token} {head_ip} {container_name}"
        )

        worker_connection.run(f"./worker_node_setup.sh {image_uri} {head_ip} {worker_ip}")

        # add timer to let container run
        time.sleep(30)

        commands = ["ray status", "fi_info -p efa"]
        for command in commands:
            head_connection.run(f"docker exec -i {container_name} /bin/bash -c '{command}'")

        serve_command = f"vllm serve {MODEL_NAME} --tensor-parallel-size 8 --pipeline-parallel-size 2 --max-num-batched-tokens 16384"
        docker_serve_command = f"docker exec -i {container_name} /bin/bash -c '{serve_command}'"
        head_connection.run(
            f"tmux new-session -d -s serve '{docker_serve_command}'", asynchronous=True
        )

        print("Waiting for model to be ready, approx estimated time to complete is 15 mins...")
        if not wait_for_container_ready(head_connection, timeout=2000):
            raise Exception("Container failed to become ready within timeout period")
        print("Model serving started successfully")

        # Run benchmark
        setup_env(head_connection)
        print("Running benchmark...")
        benchmark_cmd = "source vllm_env/bin/activate" + create_benchmark_command()
        benchmark_result = head_connection.run(benchmark_cmd, timeout=7200)
        print(f"Benchmark completed: {benchmark_result.stdout}")

        return benchmark_result

    except Exception as e:
        raise Exception(f"Multi-node test execution failed: {str(e)}")
    finally:
        head_connection.run("tmux kill-server || true", warn=True)
        worker_connection.run("tmux kill-server || true", warn=True)


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
        setup_env(connection)
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")

        setup_docker_image(connection, image_uri)
        connection.put(
            "vllm/ec2/utils/run_vllm_benchmark_single_node.sh",
            "/home/ec2-user/run_vllm_benchmark_single_node.sh",
        )
        commands = [
            "chmod +x /home/ec2-user/run_vllm_benchmark_single_node.sh",
            f"/home/ec2-user/run_vllm_benchmark_single_node.sh {image_uri} {hf_token} {MODEL_NAME}",
        ]
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
            head_conn = ec2_connections[instance_ids[0]]
            worker_conn = ec2_connections[instance_ids[1]]

            _setup_multinode_efa_instances(
                image_uri,
                resources["instances_info"][:2],
                [ec2_connections[instance_ids[0]], ec2_connections[instance_ids[1]]],
                "p4d.24xlarge",
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

            # instance_id = list(ec2_connections.keys())[0]
            # print(f"\n=== Running Single-Node Test on instance: {instance_id} ===")
            # test_results["single_node"] = run_single_node_test(
            #     ec2_connections[instance_id], image_uri
            # )

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
                    resources,
                    fsx,
                )
                cleanup_timer.cancel()
                print("Resources cleaned up successfully")
            except Exception as e:
                print(f"Cleanup failed: {str(e)}")
            finally:
                cleanup_timer.cancel()
