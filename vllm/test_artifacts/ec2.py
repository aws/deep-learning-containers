from test.test_utils.ec2 import get_account_id_from_image_uri, login_to_ecr_registry, get_ec2_client
import time, os
from vllm.infra.utils.fsx_utils import FsxSetup
from vllm.infra.ec2 import cleanup_resources

from botocore.config import Config
from fabric import Connection

DEFAULT_REGION = "us-west-2"
# HF_TOKEN = os.getenv("HF_TOKEN")

# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

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

    HF_TOKEN = get_secret_value_response["SecretString"]

    return HF_TOKEN


def test_vllm_benchmark_on_single_node(connection, image_uri):
    """
    Run VLLM benchmark test on a single node EC2 instance

    Args:
        connection: Fabric connection object to EC2 instance
        image_uri: ECR image URI for VLLM container

    Returns:
        ec2_res: Result object from test execution
    """
    try:
        # Login to ECR and pull image
        account_id = get_account_id_from_image_uri(image_uri)
        login_to_ecr_registry(connection, account_id, DEFAULT_REGION)

        print(f"Pulling image: {image_uri}")
        connection.run(f"docker pull {image_uri}", hide="out")

        # Container configuration
        container_name = "vllm-server"
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        hf_token = get_secret_hf_token()

        docker_runtime = "--runtime nvidia --gpus all"
        mount_path = "-v /fsx/.cache/huggingface:/root/.cache/huggingface"
        framework_env = f"-e HUGGING_FACE_HUB_TOKEN={hf_token} " "-e NCCL_DEBUG=TRACE"
        network_config = "-p 8000:8000 --ipc=host"

        # Start VLLM server container
        start_cmd = (
            f"docker run --name {container_name} --rm -d "
            f"{docker_runtime} {mount_path} {framework_env} {network_config} "
            f"{image_uri} --model {model_name} --tensor-parallel-size 8"
        )

        print("Starting VLLM server...")
        connection.run(start_cmd, hide=True)

        # Wait for server to be ready
        print("Waiting for server to initialize...")
        wait_cmd = """
        until $(curl --output /dev/null --silent --fail http://localhost:8000/v1/models); do
            sleep 10
        done
        """
        connection.run(wait_cmd, timeout=1800)

        # Additional delay for model loading
        time.sleep(30)

        # Run benchmark test
        test_cmd = (
            "python3 /fsx/vllm/vllm/benchmarks/benchmark_serving.py "
            f"--backend vllm --model {model_name} "
            "--endpoint /v1/completions "
            "--dataset-name sharegpt "
            "--dataset-path /fsx/vllm/ShareGPT_V3_unfiltered_cleaned_split.json "
            "--num-prompts 1000"
        )

        print("Running benchmark test...")
        ec2_res = connection.run(
            f"docker exec {container_name} bash -c '{test_cmd}'",
            hide=True,
            timeout=3600,
        )

        # Validate test results
        if ec2_res.ok:
            print("Benchmark test completed successfully")
            print(f"Test output:\n{ec2_res.stdout}")
        else:
            print(f"Benchmark test failed with error:\n{ec2_res.stderr}")

        return ec2_res

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise

    finally:
        try:
            print("Cleaning up resources...")
            connection.run(f"docker stop {container_name}", hide=True)
            connection.run(f"docker rm {container_name}", hide=True)
        except Exception as cleanup_error:
            print(f"Cleanup failed: {str(cleanup_error)}")


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


def run_vllm_test(connection, image_uri):
    """
    Main function to run VLLM benchmark test

    Args:
        connection: Fabric connection object to EC2 instance
        image_uri: ECR image URI for VLLM container
    """
    try:
        if not verify_gpu_setup(connection):
            raise Exception("GPU setup verification failed")

        print("running test_vllm_benchmark_on_single_node..")
        result = test_vllm_benchmark_on_single_node(connection, image_uri)

        if result.ok:
            print("Test completed successfully")
            return True
        else:
            print("Test failed")
            return False

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        return False


def test_vllm_on_ec2(resources, image_uri):
    """
    Test VLLM on EC2 instances with basic error handling

    Args:
        resources: Dictionary containing instance information and FSx config
        image_uri: Docker image URI to test
    """
    try:
        ec2_cli = get_ec2_client(DEFAULT_REGION)
        ec2_connections = {}
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
                ec2_connections[instance_id] = connection
                print(f"Successfully connected to instance {instance_id}")

            except Exception as e:
                print(f"Failed to connect to instance {instance_id}: {str(e)}")
                raise

        # Run tests
        try:
            for instance_id, connection in ec2_connections.items():
                print(f"Running vllm benchmarking on instance: {instance_id}")
                run_vllm_test(connection, image_uri)

        except Exception as e:
            print(f"Test execution failed: {str(e)}")
            raise

    finally:
        # Cleanup
        try:
            cleanup_resources(
                ec2_cli,
                resources["instances_info"],
                resources["sg_fsx"],
                resources["fsx_config"],
                fsx,
            )
            print("Resources cleaned up successfully")
        except Exception as e:
            print(f"Cleanup failed: {str(e)}")
