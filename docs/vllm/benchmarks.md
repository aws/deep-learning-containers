# Benchmarks

Performance benchmarks for vLLM {{ dlc_short }} images on {{ aws }} GPU instances.

## Methodology

All benchmarks use the vLLM built-in benchmarking tools with the following defaults unless noted:

- **Input/output length:** 1024 input tokens, 512 output tokens
- **Concurrency:** Saturated (max throughput)
- **Quantization:** None (FP16/BF16) unless specified
- **Warm-up:** 10 requests before measurement

## Throughput by Instance Type

### Llama 3.1 8B Instruct

| Instance Type | GPUs | TP | Throughput (tok/s) | Median TTFT (ms) | Median TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| `g5.xlarge` | 1x A10G | 1 | TBD | TBD | TBD |
| `g5.2xlarge` | 1x A10G | 1 | TBD | TBD | TBD |
| `g6.xlarge` | 1x L4 | 1 | TBD | TBD | TBD |

### Llama 3.1 70B Instruct

| Instance Type | GPUs | TP | Throughput (tok/s) | Median TTFT (ms) | Median TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| `g5.12xlarge` | 4x A10G | 4 | TBD | TBD | TBD |
| `p4d.24xlarge` | 8x A100 | 8 | TBD | TBD | TBD |
| `p5.48xlarge` | 8x H100 | 8 | TBD | TBD | TBD |

## Glossary

| Metric | Definition |
| --- | --- |
| **Throughput (tok/s)** | Total output tokens generated per second across all concurrent requests |
| **TTFT** | Time to first token — latency from request submission to first token generated |
| **TPOT** | Time per output token — average inter-token latency after the first token |
| **TP** | Tensor parallelism degree (number of GPUs) |

## Running Your Own Benchmarks

Use the vLLM built-in benchmark tool included in the {{ dlc_short }} image:

```bash
docker run --gpus all \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct &

# Run benchmark
python -m vllm.benchmarks.benchmark_serving \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 1000 \
  --request-rate inf
```

!!! note Benchmark results will be populated as new vLLM {{ dlc_short }} versions are released. Results are specific to the {{ dlc_short }} image and
may differ from upstream vLLM benchmarks due to curated patches and dependency versions.
