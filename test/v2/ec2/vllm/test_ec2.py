import boto3
import time, json
from botocore.exceptions import ClientError
from fabric import Connection

from infra.test_infra.ec2.utils import (
    get_account_id_from_image_uri,
    login_to_ecr_registry,
    install_python_in_instance,
    get_ec2_client,
)

from infra.test_infra.test_infra_utils import create_logger
from infra.test_infra.ec2.vllm.fsx_utils import FsxSetup
from infra.test_infra.ec2.vllm.setup_ec2 import TEST_ID
from test.v2.ec2.efa.test_efa import (
    _setup_multinode_efa_instances,
    EFA_SANITY_TEST_CMD,
    MASTER_CONTAINER_NAME,
    HOSTS_FILE_LOCATION,
    EFA_INTEGRATION_TEST_CMD,
    DEFAULT_EFA_TIMEOUT,
    get_efa_container_name,
)
from test.test_utils import run_cmd_on_container

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_REGION = "us-west-2"
LOGGER = create_logger(__name__)


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
    vllm bench serve \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --backend vllm \
    --base-url "http://localhost:8000" \
    --endpoint '/v1/completions' \
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


def wait_for_container_ready(connection, container_name, timeout: int = 1000) -> bool:
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
                    LOGGER.info("Model endpoint is responding")
                    LOGGER.info("\n=== Complete vLLM Server Log ===")
                    connection.run(f"docker exec {container_name} cat vllm.log", hide=False)
                    LOGGER.info("=== End of Log ===\n")
                    model_ready = True
                    return True
            except Exception:
                pass
    return False


def setup_docker_image(conn, image_uri):
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(conn, account_id, DEFAULT_REGION)
    LOGGER.info(f"Pulling image: {image_uri}")
    conn.run(f"docker pull {image_uri}", hide="out")


def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
    try:
        # Get HF token
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        if not hf_token:
            raise Exception("Failed to get HF token")

        for conn in [head_connection, worker_connection]:
            install_python_in_instance(conn, "3.10")
            setup_docker_image(conn, image_uri)
            setup_env(conn)

        head_connection.put("v2/ec2/vllm/head_node_setup.sh", "/home/ec2-user/head_node_setup.sh")
        worker_connection.put(
            "v2/ec2/vllm/worker_node_setup.sh", "/home/ec2-user/worker_node_setup.sh"
        )

        head_connection.run("chmod +x head_node_setup.sh")
        worker_connection.run("chmod +x worker_node_setup.sh")

        head_ip = head_connection.run("hostname -i").stdout.strip()
        worker_ip = worker_connection.run("hostname -i").stdout.strip()

        container_name = "ray_head-" + TEST_ID
        LOGGER.info("Starting head node...")
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
        head_connection.run(
            f"docker exec -i {container_name} /bin/bash -c '{serve_command} > vllm.log 2>&1 &'"
        )

        LOGGER.info(
            "Waiting for model to be ready, approx estimated time to complete is 15 mins..."
        )
        if not wait_for_container_ready(head_connection, container_name, timeout=2000):
            raise Exception("Container failed to become ready within timeout period")

        LOGGER.info("Running benchmark...")
        benchmark_cmd = "source vllm_env/bin/activate &&" + create_benchmark_command()
        benchmark_result = head_connection.run(benchmark_cmd, timeout=7200)

        return benchmark_result

    except Exception as e:
        raise Exception(f"Multi-node test execution failed: {str(e)}")


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
            LOGGER.info("nvidia-smi check failed")
            return False

        # Check CUDA availability
        cuda_check = connection.run("nvidia-smi -L", hide=True)
        if cuda_check.failed or "GPU" not in cuda_check.stdout:
            LOGGER.info("No GPUs found")
            return False

        return True

    except Exception as e:
        LOGGER.info(f"GPU verification failed: {str(e)}")
        return False


def cleanup_containers(connection):
    """
    Cleanup docker containers and images on the instance

    Args:
        connection: Fabric connection object
    """
    try:
        LOGGER.info("Cleaning up containers and images...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        LOGGER.error(f"Cleanup warning: {str(e)}")


def run_multi_node_test(head_conn, worker_conn, image_uri):
    """
    Run multi-node VLLM benchmark test

    Args:
        head_conn: Fabric connection object for head node
        worker_conn: Fabric connection object for worker node
        image_uri: ECR image URI
    """

    LOGGER.info("\n=== Starting Multi-Node Test ===")
    verification_tasks = [(head_conn, "head"), (worker_conn, "worker")]
    for conn, node_type in verification_tasks:
        if not verify_gpu_setup(conn):
            raise Exception(f"GPU setup verification failed for {node_type} node")

    result = test_vllm_benchmark_on_multi_node(head_conn, worker_conn, image_uri)
    if result.ok:
        LOGGER.info("Multi-node test completed successfully")
        return True
    return False


def run_single_node_test(head_conn, image_uri):
    """
    Run single-node VLLM benchmark test

    Args:
        head_conn: Fabric connection object for head node
        image_uri: ECR image URI
    """
    if not verify_gpu_setup(head_conn):
        raise Exception(f"GPU setup verification failed for head node")

    try:
        install_python_in_instance(head_conn, python_version="3.10")

        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")

        setup_docker_image(head_conn, image_uri)

        head_conn.put(
            "v2/ec2/vllm/run_vllm_on_arm64.sh",
            "/home/ec2-user/run_vllm_on_arm64.sh",
        )
        commands = [
            "chmod +x /home/ec2-user/run_vllm_on_arm64.sh",
            f"/home/ec2-user/run_vllm_on_arm64.sh {image_uri} {hf_token}",
        ]

        result = head_conn.run(
            "; ".join(commands),
            hide=False,
            timeout=7200,
        )

    except Exception as e:
        LOGGER.error(f"Test execution failed: {str(e)}")
        raise

    if result.ok:
        LOGGER.info("Single-node test completed successfully")
        return True


def test_vllm_on_ec2(resources, image_uri):
    """
    Test VLLM on EC2 instances:
    - For non-arm64: EFA testing, Single node test, and Multi-node test
    - For arm64: Single node test only

    Args:
        resources: Dictionary containing instance information and FSx config
        image_uri: Docker image URI to test

    Environment Variables:
        ARCH_TYPE: Architecture type (x86_64 or arm64)
        AWS_REGION: AWS region
        FRAMEWORK: Framework being tested (vllm)
    """
    # Read arch_type from environment variable
    import os

    arch_type = os.getenv("ARCH_TYPE", "x86_64")
    ec2_cli = None
    fsx = None
    ec2_connections = {}
    test_results = {"efa": None, "single_node": None, "multi_node": None}

    # to test agents

    try:
        ec2_cli = get_ec2_client(DEFAULT_REGION)
        fsx = FsxSetup(DEFAULT_REGION)

        # Recreate connections from stored parameters if available, otherwise create new ones
        if "connection_params" in resources and resources["connection_params"]:
            LOGGER.info("Recreating connections from stored parameters")
            # Recreate fresh Connection objects from parameters stored during setup_test_artifacts()
            for params in resources["connection_params"]:
                try:
                    connection = Connection(
                        host=params["host"],
                        user=params["user"],
                        connect_kwargs={"key_filename": params["key_filename"]},
                    )
                    ec2_connections[params["instance_id"]] = connection
                    LOGGER.info(f"Recreated connection to instance {params['instance_id']}")
                except Exception as e:
                    LOGGER.error(
                        f"Failed to recreate connection to instance {params['instance_id']}: {str(e)}"
                    )
                    raise
        else:
            LOGGER.info("Creating new connections to instances")
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

                    ec2_connections[instance_id] = connection

                except Exception as e:
                    LOGGER.error(f"Failed to connect to instance {instance_id}: {str(e)}")
                    raise

        # Verify all connections are working
        for instance_id, conn in ec2_connections.items():
            try:
                conn.run('echo "Connection test"', hide=True)
                LOGGER.info(f"Successfully verified connection to instance {instance_id}")
            except Exception as e:
                LOGGER.error(f"Connection test failed for instance {instance_id}: {str(e)}")
                raise

        is_arm64 = "arm64" in image_uri
        instance_ids = list(ec2_connections.keys())
        head_conn = ec2_connections[instance_ids[0]]

        if is_arm64:
            LOGGER.info("\n=== Starting ARM64 Single Node Test ===")
            test_results["single_node"] = run_single_node_test(head_conn, image_uri)
            LOGGER.info(
                f"ARM64 Single node test: {'Passed' if test_results['single_node'] else 'Failed'}"
            )

        elif len(ec2_connections) >= 2:
            worker_conn = ec2_connections[instance_ids[1]]

            LOGGER.info("\n=== Starting EFA Tests ===")
            _setup_multinode_efa_instances(
                image_uri,
                resources["instances_info"][:2],
                [head_conn, worker_conn],
                "p4d.24xlarge",
                DEFAULT_REGION,
                arch_type,
            )

            # Determine the master container name
            master_container_name = get_efa_container_name("vllm", "efa", arch_type, "master")

            # Run EFA sanity test
            run_cmd_on_container(master_container_name, head_conn, EFA_SANITY_TEST_CMD, hide=False)

            # Run EFA integration test
            run_cmd_on_container(
                master_container_name,
                head_conn,
                f"{EFA_INTEGRATION_TEST_CMD} {HOSTS_FILE_LOCATION} 2",
                hide=False,
                timeout=DEFAULT_EFA_TIMEOUT,
            )

            test_results["efa"] = True

            for conn in [head_conn, worker_conn]:
                cleanup_containers(conn)

            LOGGER.info("EFA tests completed successfully")

            # Run multi-node test
            test_results["multi_node"] = run_multi_node_test(head_conn, worker_conn, image_uri)

        else:
            LOGGER.info("\nSkipping multi-node test: insufficient instances")

        LOGGER.info("\n=== Test Summary ===")
        for test_name, result in test_results.items():
            if result is not None:
                LOGGER.info(
                    f"{test_name.replace('_', ' ').title()} test: {'Passed' if result else 'Failed'}"
                )
            else:
                LOGGER.info(f"{test_name.replace('_', ' ').title()} test: Not Run")

        if is_arm64:
            if not test_results["single_node"]:
                raise Exception("Single node test failed for ARM64")
        elif not any(result for result in test_results.values() if result is not None):
            raise Exception("All tests failed")

    except Exception as e:
        LOGGER.error(f"Test execution failed: {str(e)}")
        raise
