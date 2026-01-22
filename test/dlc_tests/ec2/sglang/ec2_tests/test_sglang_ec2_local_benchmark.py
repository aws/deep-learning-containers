import pytest
import time

SGLANG_EC2_GPU_INSTANCE_TYPE = "g5.2xlarge"
DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.parametrize("ec2_instance_type", [SGLANG_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_sglang_ec2_local_benchmark(sglang_inference, ec2_connection, region, gpu_only):
    """
    Test SGLang local benchmark on EC2 using ShareGPT dataset
    """
    if not gpu_only:
        pytest.skip("SGLang requires GPU")

    image_uri = sglang_inference

    # Setup dataset
    ec2_connection.run("mkdir -p /tmp/dataset")
    dataset_check = ec2_connection.run(
        "test -f /tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json && echo 'exists' || echo 'missing'",
        hide=True
    )

    if "missing" in dataset_check.stdout:
        ec2_connection.run(f"wget -P /tmp/dataset {DATASET_URL}")

    # Start container (matching your V2 workflow)
    container_cmd = f"""
    docker run -d --name sglang_benchmark --rm --gpus=all \
        -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
        -v /tmp/dataset:/dataset \
        -p 30000:30000 \
        -e HF_TOKEN=$HF_TOKEN \
        {image_uri} \
        --model-path Qwen/Qwen3-0.6B \
        --reasoning-parser qwen3 \
        --host [IP_ADDRESS] \
        --port 30000
    """

    ec2_connection.run(container_cmd)

    # Wait for startup (matching your V2 workflow)
    time.sleep(120)

    # Check logs
    ec2_connection.run("docker logs sglang_benchmark")

    # Run benchmark (matching your V2 workflow)
    benchmark_cmd = """
    docker exec sglang_benchmark python3 -m sglang.bench_serving \
        --backend sglang \
        --host [IP_ADDRESS] --port 30000 \
        --num-prompts 1000 \
        --model Qwen/Qwen3-0.6B \
        --dataset-name sharegpt \
        --dataset-path /dataset/ShareGPT_V3_unfiltered_cleaned_split.json
    """

    result = ec2_connection.run(benchmark_cmd)
    assert result.return_code == 0, f"Benchmark failed: {result.stderr}"

    # Cleanup
    ec2_connection.run("docker stop sglang_benchmark", warn=True)
