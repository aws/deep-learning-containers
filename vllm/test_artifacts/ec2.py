from test.test_utils.ec2 import get_account_id_from_image_uri, login_to_ecr_registry, get_ec2_client
import time, os, json
from vllm.infra.utils.fsx_utils import FsxSetup
from vllm.infra.ec2 import cleanup_resources

from botocore.config import Config
import threading
from fabric import Connection

DEFAULT_REGION = "us-west-2"

import boto3
from botocore.exceptions import ClientError


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


def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
    """
    Run VLLM benchmark test on multiple EC2 instances using distributed setup

    Args:
        head_connection: Fabric connection object to head EC2 instance
        worker_connection: Fabric connection object to worker EC2 instance
        image_uri: ECR image URI for VLLM container

    Returns:
        dict: Results from benchmark execution
    """
    try:
        # Get HF token
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

        # Setup ECR access on both nodes
        account_id = get_account_id_from_image_uri(image_uri)
        login_to_ecr_registry(head_connection, account_id, DEFAULT_REGION)
        login_to_ecr_registry(worker_connection, account_id, DEFAULT_REGION)

        # Pull images on both nodes
        print(f"Pulling image on head node: {image_uri}")
        head_connection.run(f"docker pull {image_uri}", hide="out")
        print(f"Pulling image on worker node: {image_uri}")
        worker_connection.run(f"docker pull {image_uri}", hide="out")

        # Get IP addresses
        head_ip = head_connection.run("hostname -i").stdout.strip()
        worker_ip = worker_connection.run("hostname -i").stdout.strip()

        # Copy cluster scripts to both nodes
        head_connection.put(
            "fsx/vllm/vllm/examples/online_serving/run_cluster.sh",
            "/home/ec2-user/run_cluster.sh",
        )
        worker_connection.put(
            "fsx/vllm/vllm/examples/online_serving/run_cluster.sh",
            "/home/ec2-user/run_cluster.sh",
        )

        # Start head node
        head_cmd = f"""
        chmod +x /home/ec2-user/run_cluster.sh &&
        /home/ec2-user/run_cluster.sh {image_uri} {head_ip} --head /fsx/.cache/huggingface \
        -e VLLM_HOST_IP={head_ip} \
        -e HF_TOKEN={hf_token} \
        -e FI_PROVIDER=efa \
        -e FI_EFA_USE_DEVICE_RDMA=1 \
        --device=/dev/infiniband/ \
        --ulimit memlock=-1:-1 \
        -p 8000:8000
        """
        head_connection.run(head_cmd, hide=False, asynchronous=True)

        time.sleep(100)

        # Start worker node
        worker_cmd = f"""
        chmod +x /home/ec2-user/run_cluster.sh &&
        /home/ec2-user/run_cluster.sh {image_uri} {head_ip} --worker /fsx/.cache/huggingface \
        -e VLLM_HOST_IP={worker_ip} \
        -e FI_PROVIDER=efa \
        -e FI_EFA_USE_DEVICE_RDMA=1 \
        --device=/dev/infiniband/ \
        --ulimit memlock=-1:-1
        """
        worker_connection.run(worker_cmd, hide=False, asynchronous=True)

        # Wait for cluster to stabilize
        time.sleep(100)

        # Execute model serving inside head node container
        serve_cmd = f"""
        docker exec node bash -c 'export HUGGING_FACE_HUB_TOKEN="{hf_token}" && \
        vllm serve {model_name} \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --max-num-batched-tokens 16384'
        """
        head_connection.run(serve_cmd, hide=False, asynchronous=True)

        time.sleep(300)

        # Run benchmark
        benchmark_cmd = f"""
        conda activate vllm && \
        python3 /fsx/vllm/vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model {model_name} \
        --endpoint /v1/completions \
        --dataset-name sharegpt \
        --dataset-path /fsx/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000
        """
        result = head_connection.run(benchmark_cmd, hide=False, timeout=7200)

        return result

    except Exception as e:
        print(f"Multi-node test execution failed: {str(e)}")
        raise


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
        print("HF_TOKEN", hf_token)
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
        if result.ok:
            print("Multi-node test completed successfully")
            return True
        return False

    finally:
        for conn in [head_conn, worker_conn]:
            cleanup_containers(conn)


def test_vllm_on_ec2(resources, image_uri):
    """
    Test VLLM on EC2 instances sequentially - single node followed by multi-node

    Args:
        resources: Dictionary containing instance information and FSx config
        image_uri: Docker image URI to test
    """
    ec2_cli = None
    fsx = None
    ec2_connections = {}
    test_results = {"single_node": False, "multi_node": False}

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

        # # Run single-node test on first instance
        # instance_id = list(ec2_connections.keys())[0]
        # print(f"\nRunning single-node test on instance: {instance_id}")
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
        print(f"Single-node test: {'Passed' if test_results['single_node'] else 'Failed'}")
        print(f"Multi-node test: {'Passed' if test_results['multi_node'] else 'Failed'}")

        if not any(test_results.values()):
            raise Exception("All tests failed")

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise

    finally:
        if ec2_cli and fsx:
            cleanup_timer = threading.Timer(
                300, lambda: print("Cleanup timed out, some resources might need manual cleanup")
            )
            cleanup_timer.start()

            try:
                cleanup_resources(
                    ec2_cli,
                    resources["instances_info"],
                    resources["sg_fsx"],
                    resources["fsx_config"],
                    fsx,
                )
                cleanup_timer.cancel()
                print("Resources cleaned up successfully")
            except Exception as e:
                print(f"Cleanup failed: {str(e)}")
            finally:
                cleanup_timer.cancel()
