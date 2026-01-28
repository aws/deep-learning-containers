"""
SGLang EC2 Tests
Tests SGLang inference capabilities on EC2 instances
"""

import json
import logging
import os
import sys
import time

import boto3
import pytest
from botocore.exceptions import ClientError

from test.test_utils import get_account_id_from_image_uri
from test.test_utils.ec2 import login_to_ecr_registry

# Setup logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)

# Test constants
SGLANG_EC2_GPU_INSTANCE_TYPE = "g5.2xlarge"
SGLANG_EC2_LARGE_GPU_INSTANCE_TYPE = "g6e.xlarge"
SGLANG_VERSION = "0.5.6"
DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_REGION = "us-west-2"


def get_hf_token_from_secrets_manager():
    """
    Retrieve HuggingFace token from AWS Secrets Manager

    Returns:
        str: HuggingFace token or empty string if not found
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
        LOGGER.warning(f"Failed to retrieve HF_TOKEN from Secrets Manager: {e}")
        return ""
    except Exception as e:
        LOGGER.warning(f"Unexpected error retrieving HF_TOKEN: {e}")
        return ""


def setup_docker_image(ec2_connection, image_uri):
    """
    Pull SGLang Docker image from ECR

    Args:
        ec2_connection: Fabric connection to EC2 instance
        image_uri: Docker image URI
    """
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(ec2_connection, account_id, DEFAULT_REGION)
    LOGGER.info(f"Pulling SGLang image: {image_uri}")
    ec2_connection.run(f"docker pull {image_uri}", hide="out")


def setup_dataset(ec2_connection):
    """
    Download ShareGPT dataset for benchmarking

    Args:
        ec2_connection: Fabric connection to EC2 instance
    """
    ec2_connection.run("mkdir -p /tmp/dataset")

    dataset_check = ec2_connection.run(
        "test -f /tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json && echo 'exists' || echo 'missing'",
        hide=True,
    )

    if "missing" in dataset_check.stdout:
        LOGGER.info("Downloading ShareGPT dataset...")
        ec2_connection.run(f"wget -P /tmp/dataset {DATASET_URL}")
    else:
        LOGGER.info("ShareGPT dataset already exists, skipping download")


def cleanup_containers(ec2_connection):
    """
    Cleanup all Docker containers

    Args:
        ec2_connection: Fabric connection to EC2 instance
    """
    try:
        LOGGER.info("Cleaning up Docker containers...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            ec2_connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        LOGGER.warning(f"Cleanup warning: {e}")


@pytest.mark.model("Qwen3-0.6B")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_local_benchmark(ec2_connection, sglang):
    """
    Test SGLang local benchmark on EC2 using ShareGPT dataset

    This test validates:
    - SGLang server startup with Qwen3-0.6B model
    - Benchmark execution with 1000 prompts
    - Basic inference capabilities

    Args:
        ec2_connection: Fabric connection to EC2 instance
        sglang: SGLang Docker image URI
    """
    try:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Starting SGLang EC2 Local Benchmark Test")
        LOGGER.info("=" * 80 + "\n")

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

        LOGGER.info("Starting SGLang server container...")
        ec2_connection.run(container_cmd)

        # Wait for server startup
        LOGGER.info("Waiting for server startup (120s)...")
        time.sleep(120)

        # Check container logs
        LOGGER.info("Container logs:")
        ec2_connection.run(f"docker logs {container_name}")

        # Run benchmark
        LOGGER.info("Running SGLang benchmark...")
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
            LOGGER.info("\n✓ SGLang local benchmark test passed successfully")
        else:
            LOGGER.error(f"\n✗ Benchmark test failed with return code {result.return_code}")
            raise AssertionError(f"Benchmark test failed with return code {result.return_code}")

    finally:
        cleanup_containers(ec2_connection)


@pytest.mark.model("Llama-3.1-8B")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_LARGE_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_upstream(ec2_connection, sglang):
    """
    Test SGLang upstream test suite on EC2

    This test validates:
    - Compatibility with SGLang upstream test suite (stage-a-test-1)
    - Support for gated models (Llama-3.1-8B)
    - Test execution in containerized environment

    Note: Uses g6e.xlarge (1x L40S 48GB GPU) to accommodate Llama-3.1-8B model

    Args:
        ec2_connection: Fabric connection to EC2 instance
        sglang: SGLang Docker image URI
    """
    try:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("Starting SGLang EC2 Upstream Test")
        LOGGER.info("=" * 80 + "\n")

        # Setup
        setup_docker_image(ec2_connection, sglang)

        # Get HuggingFace token from AWS Secrets Manager
        LOGGER.info("Retrieving HF_TOKEN from AWS Secrets Manager...")
        hf_token = get_hf_token_from_secrets_manager()

        if not hf_token:
            # Fallback to environment variable
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                LOGGER.info("Using HF_TOKEN from environment variable")
            else:
                pytest.skip(
                    "HF_TOKEN not found in Secrets Manager or environment. Skipping test requiring gated models."
                )

        # Clone SGLang source
        LOGGER.info("Cloning SGLang source repository...")
        ec2_connection.run("rm -rf /tmp/sglang_source", warn=True)
        ec2_connection.run(
            f"git clone --branch v{SGLANG_VERSION} --depth 1 "
            f"https://github.com/sgl-project/sglang.git /tmp/sglang_source"
        )

        # Start container with bash entrypoint
        container_name = "sglang_upstream"
        container_cmd = f"""
        docker run -d --name {container_name} --rm --gpus=all \
            --user root \
            --entrypoint /bin/bash \
            -v /home/ec2-user/.cache/huggingface:/root/.cache/huggingface \
            -v /tmp/sglang_source:/workdir \
            --workdir /workdir \
            -e HF_TOKEN={hf_token} \
            -e HUGGINGFACE_HUB_TOKEN={hf_token} \
            {sglang} \
            -c "tail -f /dev/null"
        """

        LOGGER.info("Starting SGLang container with bash entrypoint...")
        ec2_connection.run(container_cmd)

        # Verify HF token is available
        LOGGER.info("Verifying HuggingFace token...")
        ec2_connection.run(
            f"""docker exec -u root {container_name} bash -c '
                env | grep -E "HF_TOKEN|HUGGINGFACE_HUB_TOKEN" || echo "No HF tokens found"
            '""",
            warn=True,
        )

        # Install test dependencies
        LOGGER.info("Installing SGLang test dependencies...")
        ec2_connection.run(
            f"docker exec -u root {container_name} bash scripts/ci/ci_install_dependency.sh"
        )

        # Authenticate with HuggingFace for gated models
        LOGGER.info("Authenticating with HuggingFace for gated model access...")
        ec2_connection.run(
            f"docker exec -u root {container_name} huggingface-cli login --token {hf_token}"
        )
        LOGGER.info("✓ Successfully authenticated with HuggingFace")

        # Check GPU availability
        LOGGER.info("Checking GPU availability:")
        ec2_connection.run(f"docker exec -u root {container_name} nvidia-smi")

        # Run upstream test suite
        LOGGER.info("Running SGLang upstream test suite (stage-a-test-1)...")
        test_cmd = f"""
        docker exec -u root {container_name} sh -c '
            set -eux
            cd /workdir/test
            python3 run_suite.py --hw cuda --suite stage-a-test-1
        '
        """

        result = ec2_connection.run(test_cmd)

        # Capture logs if test fails
        if result.return_code != 0:
            LOGGER.error("Capturing container logs for debugging...")
            ec2_connection.run(f"docker logs {container_name} --tail 200", warn=True)

        if result.return_code == 0:
            LOGGER.info("\n✓ SGLang upstream test passed successfully")
        else:
            LOGGER.error(f"\n✗ Upstream test failed with return code {result.return_code}")
            raise AssertionError(f"Upstream test failed with return code {result.return_code}")

    finally:
        cleanup_containers(ec2_connection)
        # Run as sudo since files may have been created by root in container
        ec2_connection.run("sudo rm -rf /tmp/sglang_source", warn=True)
