import pytest

SGLANG_EC2_GPU_INSTANCE_TYPE = "g5.12xlarge"
SGLANG_VERSION = "0.5.6"

@pytest.mark.model("meta-llama/Llama-3.2-1B")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_upstream(sglang_inference, ec2_connection, region, gpu_only):
    """
    Test SGLang upstream test suite on EC2
    """
    if not gpu_only:
        pytest.skip("SGLang requires GPU")

    image_uri = sglang_inference

    # Clone SGLang source (matching your V2 workflow)
    ec2_connection.run("rm -rf /tmp/sglang_source", warn=True)
    ec2_connection.run(
        f"git clone --branch v{SGLANG_VERSION} --depth 1 "
        f"https://github.com/sgl-project/sglang.git /tmp/sglang_source"
    )

    # Start container (matching your V2 workflow)
    container_cmd = f"""
    docker run -d --name sglang_upstream --rm --gpus=all \
        --entrypoint /bin/bash \
        -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
        -v /tmp/sglang_source:/workdir \
        --workdir /workdir \
        -e HF_TOKEN=$HF_TOKEN \
        {image_uri}
    """

    ec2_connection.run(container_cmd)

    # Install dependencies (matching your V2 workflow)
    ec2_connection.run(
        "docker exec sglang_upstream bash scripts/ci/ci_install_dependency.sh"
    )

    # Check GPU
    ec2_connection.run("docker exec sglang_upstream nvidia-smi")

    # Run tests (matching your V2 workflow)
    test_cmd = """
    docker exec sglang_upstream sh -c '
        set -eux
        cd /workdir/test
        python3 run_suite.py --hw cuda --suite stage-a-test-1
    '
    """

    result = ec2_connection.run(test_cmd)
    assert result.return_code == 0, f"Upstream test failed: {result.stderr}"

    # Cleanup
    ec2_connection.run("docker stop sglang_upstream", warn=True)
    ec2_connection.run("rm -rf /tmp/sglang_source", warn=True)

