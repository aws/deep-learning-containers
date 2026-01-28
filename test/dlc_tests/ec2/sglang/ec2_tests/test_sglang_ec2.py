"""
SGLang EC2 Tests
Tests SGLang inference capabilities on EC2 instances
"""

import time
import os
import json
import boto3
import pytest
from botocore.exceptions import ClientError
from test.test_utils.ec2 import (
    get_account_id_from_image_uri,
    login_to_ecr_registry,
)

# Test constants
SGLANG_EC2_GPU_INSTANCE_TYPE = "g5.2xlarge"
SGLANG_EC2_LARGE_GPU_INSTANCE_TYPE = "g5.12xlarge"
SGLANG_VERSION = "0.5.6"
DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_REGION = "us-west-2"


def get_hf_token_from_secrets_manager():
    """
    Retrieve HuggingFace token from AWS Secrets Manager
    Same approach as v2 infrastructure
    """
    secret_name = "test/hf_token"
    region_name = "us-west-2"
    
    try:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        response = json.loads(get_secret_value_response["SecretString"])
        hf_token = response.get("HF_TOKEN", "")
        return hf_token
    except ClientError as e:
        print(f"Warning: Failed to retrieve HF_TOKEN from Secrets Manager: {e}")
        return ""
    except Exception as e:
        print(f"Warning: Unexpected error retrieving HF_TOKEN: {e}")
        return ""


def setup_docker_image(ec2_connection, image_uri):
    """Pull SGLang Docker image from ECR"""
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(ec2_connection, account_id, DEFAULT_REGION)
    print(f"Pulling SGLang image: {image_uri}")
    ec2_connection.run(f"docker pull {image_uri}", hide="out")


def setup_dataset(ec2_connection):
    """Download ShareGPT dataset for benchmarking"""
    ec2_connection.run("mkdir -p /tmp/dataset")

    dataset_check = ec2_connection.run(
        "test -f /tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json && echo 'exists' || echo 'missing'",
        hide=True,
    )

    if "missing" in dataset_check.stdout:
        print("Downloading ShareGPT dataset...")
        ec2_connection.run(f"wget -P /tmp/dataset {DATASET_URL}")
    else:
        print("ShareGPT dataset already exists. Skipping download.")


def cleanup_containers(ec2_connection):
    """Cleanup all Docker containers"""
    try:
        print("Cleaning up containers...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            ec2_connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        print(f"Cleanup warning: {e}")


@pytest.mark.model("Qwen3-0.6B")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_local_benchmark(ec2_connection, sglang):
    """
    Test SGLang local benchmark on EC2 using ShareGPT dataset

    This test:
    1. Downloads ShareGPT dataset
    2. Starts SGLang server with Qwen3-0.6B model
    3. Runs benchmark with 1000 prompts
    4. Validates successful completion
    """
    try:
        print("\n" + "=" * 80)
        print("Starting SGLang EC2 Local Benchmark Test")
        print("=" * 80 + "\n")

        # Setup
        setup_docker_image(ec2_connection, sglang)
        setup_dataset(ec2_connection)

        # Get HuggingFace token
        hf_token = os.environ.get("HF_TOKEN", "")

        # Start SGLang container
        container_name = "sglang_benchmark"
        container_cmd = f"""
        docker run -d --name {container_name} --rm --gpus=all \
            -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
            -v /tmp/dataset:/dataset \
            -p 30000:30000 \
            -e HF_TOKEN={hf_token} \
            {sglang} \
            --model-path Qwen/Qwen3-0.6B \
            --reasoning-parser qwen3 \
            --host 0.0.0.0 \
            --port 30000
        """

        print("Starting SGLang server container...")
        ec2_connection.run(container_cmd)

        # Wait for server startup
        print("Waiting for serving endpoint startup (120s)...")
        time.sleep(120)

        # Check container logs
        print("\nContainer logs:")
        ec2_connection.run(f"docker logs {container_name}")

        # Run benchmark
        print("\nRunning SGLang benchmark...")
        benchmark_cmd = f"""
        docker exec {container_name} python3 -m sglang.bench_serving \
            --backend sglang \
            --host 0.0.0.0 --port 30000 \
            --num-prompts 1000 \
            --model Qwen/Qwen3-0.6B \
            --dataset-name sharegpt \
            --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
        """

        result = ec2_connection.run(benchmark_cmd)

        if result.return_code == 0:
            print("\n✓ SGLang local benchmark test passed successfully")
        else:
            print(f"\n✗ Benchmark test failed with return code {result.return_code}")
            raise AssertionError(f"Benchmark test failed with return code {result.return_code}")

    finally:
        cleanup_containers(ec2_connection)


@pytest.mark.model("Qwen3-0.6B")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_LARGE_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_upstream(ec2_connection, sglang):
    """
    Test SGLang upstream test suite on EC2

    This test:
    1. Clones SGLang upstream repository (v0.5.6)
    2. Starts SGLang container with bash entrypoint
    3. Installs test dependencies
    4. Authenticates with HuggingFace for gated models
    5. Runs upstream test suite (stage-a-test-1)
    6. Validates test results
    """
    try:
        print("\n" + "=" * 80)
        print("Starting SGLang EC2 Upstream Test")
        print("=" * 80 + "\n")

        # Setup
        setup_docker_image(ec2_connection, sglang)

        # Get HuggingFace token from AWS Secrets Manager (same as v2 infrastructure)
        print("Retrieving HF_TOKEN from AWS Secrets Manager...")
        hf_token = get_hf_token_from_secrets_manager()
        
        if not hf_token:
            # Fallback to environment variable
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                print("Using HF_TOKEN from environment variable")
            else:
                pytest.skip("HF_TOKEN not found in Secrets Manager or environment. Skipping test requiring gated models.")

        # Clone SGLang source
        print("Cloning SGLang source repository...")
        ec2_connection.run("rm -rf /tmp/sglang_source", warn=True)
        ec2_connection.run(
            f"git clone --branch v{SGLANG_VERSION} --depth 1 "
            f"https://github.com/sgl-project/sglang.git /tmp/sglang_source"
        )

        # Start container with bash entrypoint
        container_name = "sglang_upstream"
        container_cmd = f"""
        docker run -d --name {container_name} --rm --gpus=all \
            --entrypoint /bin/bash \
            -v /home/ec2-user/.cache/huggingface:/root/.cache/huggingface \
            -v /tmp/sglang_source:/workdir \
            --workdir /workdir \
            -e HF_TOKEN={hf_token} \
            {sglang} \
            -c "tail -f /dev/null"
        """

        print("Starting SGLang container with bash entrypoint...")
        ec2_connection.run(container_cmd)

        # Authenticate with HuggingFace BEFORE installing dependencies
        # This ensures gated models are accessible during test setup
        print("\nAuthenticating with HuggingFace...")

        # Login with token (this saves credentials to ~/.cache/huggingface/token)
        login_result = ec2_connection.run(
            f"docker exec {container_name} huggingface-cli login --token {hf_token}", warn=True
        )

        if login_result.return_code != 0:
            raise AssertionError(
                f"Failed to authenticate with HuggingFace. "
                f"Return code: {login_result.return_code}, "
                f"Output: {login_result.stdout}"
            )

        # Verify authentication worked
        auth_check = ec2_connection.run(
            f"docker exec {container_name} huggingface-cli whoami", warn=True
        )

        if auth_check.return_code == 0:
            print(f"✓ Successfully authenticated as: {auth_check.stdout.strip()}")
        else:
            raise AssertionError(
                f"HuggingFace authentication verification failed. "
                f"The token may be invalid or expired. "
                f"Return code: {auth_check.return_code}"
            )

        # Test access to a gated model to ensure permissions are working
        print("\nVerifying access to gated models...")
        test_access = ec2_connection.run(
            f'docker exec {container_name} python3 -c "'
            f"from huggingface_hub import HfApi; "
            f"api = HfApi(); "
            f'info = api.model_info(\\"meta-llama/Llama-3.1-8B-Instruct\\"); '
            f'print(f\\"✓ Can access gated model: {{info.id}}\\")'
            f'"',
            warn=True,
        )

        if test_access.return_code != 0:
            print("⚠ Warning: Cannot access meta-llama/Llama-3.1-8B-Instruct")
            print("This may cause upstream tests to fail if they require this model")
            print(f"Error output: {test_access.stderr}")
        else:
            print(test_access.stdout.strip())

        # Install test dependencies
        print("\nInstalling SGLang test dependencies...")
        ec2_connection.run(f"docker exec {container_name} bash scripts/ci/ci_install_dependency.sh")

        # Check GPU availability
        print("\nChecking GPU availability:")
        ec2_connection.run(f"docker exec {container_name} nvidia-smi")

        # Run upstream test suite
        print("\nRunning SGLang upstream test suite (stage-a-test-1)...")
        test_cmd = f"""
        docker exec {container_name} sh -c '
            set -eux
            cd /workdir/test
            python3 run_suite.py --hw cuda --suite stage-a-test-1
        '
        """

        result = ec2_connection.run(test_cmd)

        if result.return_code == 0:
            print("\n✓ SGLang upstream test passed successfully")
        else:
            print(f"\n✗ Upstream test failed with return code {result.return_code}")
            raise AssertionError(f"Upstream test failed with return code {result.return_code}")

    finally:
        cleanup_containers(ec2_connection)
        ec2_connection.run("rm -rf /tmp/sglang_source", warn=True)
