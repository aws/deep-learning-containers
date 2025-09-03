#arm64 testing

set -e

DLC_IMAGE=$1
HF_TOKEN=$2

docker run -v $(pwd):/vllm \
  -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /fsx/.cache/huggingface:/root/.cache/huggingface \
  --name vllm-arm64-dlc \
  --gpus=all \
  --entrypoint="" \
  $DLC_IMAGE \
  bash -c 'python3 /vllm/examples/offline_inference/basic/generate.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype half \
    --tensor-parallel-size 1 \
    --max-model-len 2048'


