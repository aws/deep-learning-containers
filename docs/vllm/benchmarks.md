# Benchmarks

Performance benchmarks for vLLM {{ dlc_short }} images on {{ aws }} GPU instances.

## Methodology

All benchmarks use the vLLM built-in benchmarking tools (`vllm bench throughput` and `vllm bench latency`) with the following defaults unless noted:

- **Dataset:** Random tokens
- **Quantization:** None (BF16) unless specified
- **Warm-up:** 3 iterations before measurement

## Throughput Results

Results from CI benchmark runs on {{ aws }} GPU instances.

### CodeBuild Fleet

| Model | Instance | GPUs | TP | Input Len | Output Len | Prompts | Min Tokens/s | Min Rps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS 20B | g6e.xlarge (1x L40S) | 1 | 1 | 512 | 128 | 64 | 6000 | 5 |
| Qwen 3.5 9B | g6.xlarge (1x L4) | 1 | 1 | 512 | 128 | 64 | 180 | 0.15 |
| Llama 3.3 70B | g6e.12xlarge (4x L40S) | 4 | 4 | 512 | 128 | 32 | 400 | 0.35 |
| Qwen 3.5 35B-A3B FP8 | g6e.12xlarge (4x L40S) | 4 | 4 | 512 | 128 | 64 | 400 | 0.35 |
| Qwen 3.5 27B FP8 | g6e.12xlarge (4x L40S) | 4 | 4 | 512 | 128 | 64 | 100 | 0.2 |
| Qwen 3 Coder Next FP8 | g6e.12xlarge (4x L40S) | 4 | 4 | 512 | 256 | 32 | 280 | 0.25 |

### Runner Scale Sets (A100)

| Model | GPUs | TP | Input Len | Output Len | Prompts | Min Tokens/s | Min Rps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 3 32B | 4x A100 40 GB | 4 | 512 | 256 | 32 | 3400 | 3 |
| GPT-OSS 20B | 1x A100 40 GB | 1 | 512 | 128 | 64 | 6000 | 5 |
| Qwen 3.5 35B-A3B FP8 | 4x A100 40 GB | 4 | 512 | 128 | 64 | 400 | 0.35 |
| Qwen 3.5 27B FP8 | 4x A100 40 GB | 4 | 512 | 128 | 64 | 100 | 0.2 |
| Qwen 3 Coder Next FP8 | 4x A100 40 GB | 4 | 512 | 256 | 32 | 280 | 0.25 |
| Llama 3.3 70B | 4x A100 40 GB | 4 | 512 | 128 | 32 | 400 | 0.35 |

!!! note FP8 models on A100 (compute capability 8.0) use Marlin dequantization fallback since native FP8 requires compute capability 8.9+ (H100/L40S).
This increases memory usage and reduces throughput compared to L40S/H100.

## Glossary

| Metric | Definition |
| --- | --- |
| **Tokens/s** | Total output tokens generated per second across all concurrent requests |
| **Rps** | Requests per second completed |
| **TP** | Tensor parallelism degree (number of GPUs) |
| **Min Tokens/s** | Minimum throughput threshold for CI pass |
| **Min Rps** | Minimum requests/s threshold for CI pass |

## Running Your Own Benchmarks

Use the vLLM built-in benchmark tools included in the {{ dlc_short }} image:

```bash
docker run --gpus all -v /local/models:/models \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  vllm bench throughput \
    --model /models/my-model \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 64
```

```bash
docker run --gpus all -v /local/models:/models \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  vllm bench latency \
    --model /models/my-model \
    --input-len 512 \
    --output-len 128 \
    --batch-size 4 \
    --num-iters 10
```

!!! note Benchmark results reflect CI minimum thresholds and may vary based on instance configuration, model version, and vLLM version. Run your own
benchmarks for production capacity planning.
