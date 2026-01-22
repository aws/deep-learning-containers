"""
SGLang EC2 Tests
Tests SGLang inference capabilities on EC2 instances
"""

import time
import os
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


def setup_docker_image(connection, image_uri):
    """Pull SGLang Docker image from ECR"""
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(connection, account_id, DEFAULT_REGION)
    print(f"Pulling SGLang image: {image_uri}")
    connection.run(f"docker pull {image_uri}", hide="out")


def setup_dataset(connection):
    """Download ShareGPT dataset for benchmarking"""
    connection.run("mkdir -p /tmp/dataset")

    dataset_check = connection.run(
        "test -f /tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json && echo 'exists' || echo 'missing'",
        hide=True,
    )

    if "missing" in dataset_check.stdout:
        print("Downloading ShareGPT dataset...")
        connection.run(f"wget -P /tmp/dataset {DATASET_URL}")
    else:
        print("ShareGPT dataset already exists. Skipping download.")


def cleanup_containers(connection):
    """Cleanup all Docker containers"""
    try:
        print("Cleaning up containers...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        print(f"Cleanup warning: {e}")


def test_sglang_ec2_local_benchmark(connection, image_uri):
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
        setup_docker_image(connection, image_uri)
        setup_dataset(connection)

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
            {image_uri} \
            --model-path Qwen/Qwen3-0.6B \
            --reasoning-parser qwen3 \
            --host 0.0.0.0 \
            --port 30000
        """

        print("Starting SGLang server container...")
        connection.run(container_cmd)

        # Wait for server startup
        print("Waiting for serving endpoint startup (120s)...")
        time.sleep(120)

        # Check container logs
        print("\nContainer logs:")
        connection.run(f"docker logs {container_name}")

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

        result = connection.run(benchmark_cmd)

        if result.return_code == 0:
            print("\n✓ SGLang local benchmark test passed successfully")
            return True
        else:
            print(f"\n✗ Benchmark test failed with return code {result.return_code}")
            return False

    except Exception as e:
        print(f"\nLocal benchmark test failed: {str(e)}")
        return False
    finally:
        cleanup_containers(connection)


def test_sglang_ec2_upstream(connection, image_uri):
    """
    Test SGLang upstream test suite on EC2

    This test:
    1. Clones SGLang upstream repository (v0.5.6)
    2. Starts SGLang container with bash entrypoint
    3. Installs test dependencies
    4. Runs upstream test suite (stage-a-test-1)
    5. Validates test results
    """
    try:
        print("\n" + "=" * 80)
        print("Starting SGLang EC2 Upstream Test")
        print("=" * 80 + "\n")

        # Setup
        setup_docker_image(connection, image_uri)

        # Get HuggingFace token
        hf_token = os.environ.get("HF_TOKEN", "")

        # Clone SGLang source
        print("Cloning SGLang source repository...")
        connection.run("rm -rf /tmp/sglang_source", warn=True)
        connection.run(
            f"git clone --branch v{SGLANG_VERSION} --depth 1 "
            f"https://github.com/sgl-project/sglang.git /tmp/sglang_source"
        )

        # Start container with bash entrypoint
        container_name = "sglang_upstream"
        container_cmd = f"""
        docker run -d --name {container_name} --rm --gpus=all \
            --entrypoint /bin/bash \
            -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
            -v /tmp/sglang_source:/workdir \
            --workdir /workdir \
            -e HF_TOKEN={hf_token} \
            {image_uri}
        """

        print("Starting SGLang container with bash entrypoint...")
        connection.run(container_cmd)

        # Install test dependencies
        print("\nInstalling SGLang test dependencies...")
        connection.run(f"docker exec {container_name} bash scripts/ci/ci_install_dependency.sh")

        # Check GPU availability
        print("\nChecking GPU availability:")
        connection.run(f"docker exec {container_name} nvidia-smi")

        # Run upstream test suite
        print("\nRunning SGLang upstream test suite (stage-a-test-1)...")
        test_cmd = f"""
        docker exec {container_name} sh -c '
            set -eux
            cd /workdir/test
            python3 run_suite.py --hw cuda --suite stage-a-test-1
        '
        """

        result = connection.run(test_cmd)

        if result.return_code == 0:
            print("\n✓ SGLang upstream test passed successfully")
            return True
        else:
            print(f"\n✗ Upstream test failed with return code {result.return_code}")
            return False

    except Exception as e:
        print(f"\nUpstream test failed: {str(e)}")
        return False
    finally:
        cleanup_containers(connection)
        connection.run("rm -rf /tmp/sglang_source", warn=True)


def test_sglang_on_ec2(resources, image_uri):
    """
    Main test orchestrator for SGLang EC2 tests

    Runs both local benchmark and upstream tests on EC2 instances

    Args:
        resources: Dictionary containing instance information
        image_uri: SGLang Docker image URI to test
    """
    from fabric import Connection

    ec2_connections = {}
    test_results = {"local_benchmark": None, "upstream": None}

    try:
        # Setup connections to EC2 instances
        for instance_id, key_filename in resources["instances_info"]:
            try:
                # Get instance public IP
                public_ip = resources.get("public_ips", {}).get(instance_id)

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

        # Get primary connection
        instance_ids = list(ec2_connections.keys())
        head_conn = ec2_connections[instance_ids[0]]

        # Run tests
        print("\n" + "=" * 80)
        print("Starting SGLang EC2 Tests")
        print("=" * 80)

        # Test 1: Local Benchmark
        test_results["local_benchmark"] = test_sglang_ec2_local_benchmark(head_conn, image_uri)

        # Test 2: Upstream Tests
        test_results["upstream"] = test_sglang_ec2_upstream(head_conn, image_uri)

        # Print summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        for test_name, result in test_results.items():
            if result is not None:
                status = "✓ Passed" if result else "✗ Failed"
                print(f"{test_name.replace('_', ' ').title()}: {status}")
            else:
                print(f"{test_name.replace('_', ' ').title()}: Not Run")

        # Check if any test failed
        if not all(result for result in test_results.values() if result is not None):
            raise Exception("One or more SGLang tests failed")

        print("\n✓ All SGLang EC2 tests passed successfully")

    except Exception as e:
        print(f"\nTest execution failed: {str(e)}")
        raise
