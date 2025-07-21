from test.test_utils.ec2 import get_account_id_from_image_uri, login_to_ecr_registry, get_ec2_client
import time, os, json
from vllm.infra.utils.fsx_utils import FsxSetup
from vllm.infra.ec2 import cleanup_resources

from botocore.config import Config
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
