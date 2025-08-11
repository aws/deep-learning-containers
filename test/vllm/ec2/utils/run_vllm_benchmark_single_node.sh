set -e


CONTAINER_IMAGE=$1
HF_TOKEN=$2
MODEL_NAME=$3

INSTANCE_TYPE="p4d.24xlarge"
UPSTREAM_IMAGE="vllm/vllm-openai:latest"

# Function to run benchmark
run_benchmark() {
    local container_image=$1
    local output_file=$2
    local container_name="vllm-server-${RANDOM}"

    echo "Running benchmark for $container_image"
    docker run --entrypoint=/bin/bash --name $container_name --rm -d --runtime nvidia --gpus all \
        -v /fsx/.cache/huggingface:/root/.cache/huggingface \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "NCCL_DEBUG=TRACE" \
        -p 8000:8000 \
        --ipc=host \
        $container_image \
        --model ${MODEL_NAME} \
        --tensor-parallel-size 8
        
    echo "Waiting for model to fully load..."
    sleep 1000

    echo "Running benchmark..."
    python3 /fsx/vllm-dlc/benchmarks/benchmark_serving.py \
      --backend vllm \
      --model ${MODEL_NAME} \
      --endpoint /v1/completions \
      --dataset-name sharegpt \
      --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 1000 > $output_file

    # Stop and remove the docker container
    docker stop $container_name
    docker rm $container_name
}

# Run benchmarks
run_benchmark "$CONTAINER_IMAGE" "dlc_benchmark.txt"

echo "vLLM DLC results:"
echo "--------------------"
cat dlc_benchmark.txt
echo "--------------------"

