# Benchmarks

## Methodology

All benchmarks use the SGLang built-in benchmarking tool (`sglang.bench_serving`) with the following defaults:

- **Dataset:** ShareGPT (realistic conversation distribution)
- **Prompts:** 1000
- **Concurrency:** Saturated (max throughput)
- **Quantization:** None (BF16) unless specified as FP8

## Throughput by Model

| Model | Instance Type | TP | Output tok/s | TTFT (ms) | TPOT (ms) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | `g6e.xlarge` | 1 | TBD | TBD | TBD |
| GPT-OSS-20B | `g6e.xlarge` | 1 | TBD | TBD | TBD |
| Qwen3-32B | `p4d.24xlarge` / `gpu-efa-runners` | 4 | TBD | TBD | TBD |
| Qwen3.5-27B-FP8 | `g6e.xlarge` | 1 | TBD | TBD | TBD |

> **Note:** Benchmark data will be published after GA release.

## Latency by Model

| Model | Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | `g6e.xlarge` | 1 | TBD | TBD | TBD | TBD | TBD |
| GPT-OSS-20B | `g6e.xlarge` | 1 | TBD | TBD | TBD | TBD | TBD |
| Qwen3-32B | `p4d.24xlarge` / `gpu-efa-runners` | 4 | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-27B-FP8 | `g6e.xlarge` | 1 | TBD | TBD | TBD | TBD | TBD |

> **Note:** Latency data will be published after GA release.

## Glossary

| Metric | Definition |
| --- | --- |
| **Output tok/s** | Output tokens generated per second across all concurrent requests |
| **TTFT** | Time to first token — latency from request submission to first token generated |
| **TPOT** | Time per output token — average inter-token latency after the first token |
| **TP** | Tensor parallelism degree (number of GPUs) |

## Running Your Own Benchmarks

```bash
# Start the server
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000

# In another terminal, run the benchmark
docker exec <container_id> python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json
```

> **Note:** Results are specific to the {{ dlc_short }} image and may differ from upstream SGLang benchmarks due to curated patches and dependency
> versions.

## See Also

- [Configuration](configuration.md) — memory and performance tuning
- [Deployment](deployment.md) — recommended instance types
- [Supported Models](supported_models.md) — tested models and compatibility
