# Benchmarks

Performance benchmarks for vLLM {{ dlc_short }} images on {{ aws }} GPU instances.

## Methodology

All benchmarks use the vLLM built-in benchmarking tools with the following defaults unless noted:

- **Input length:** 1024 tokens
- **Output length:** 128 tokens
- **Concurrency:** Saturated (max throughput)
- **Quantization:** None (FP16/BF16) unless specified as FP8
- **Warm-up:** 10 requests before measurement
- **Throughput metric:** Output tokens per second (generated tokens only, excludes prompt tokens)

## Throughput by Model

### Qwen3.5-9B

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `g6.xlarge` | 1 | 1024 / 128 | 64 | 22.77 | 0.18 |

> **Note — Eager mode:** `--enforce-eager` is enabled due to a vLLM CUDA graph incompatibility with this model's hybrid architecture. Performance is
> significantly lower than expected.

### GPT-OSS-20B

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `g6e.xlarge` | 1 | 1024 / 128 | 64 | 1,393.03 | 10.88 |

### Llama 3.3 70B Instruct

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 1024 / 128 | 32 | 215.54 | 1.68 |
| `g6e.12xlarge` | 4 | 1024 / 128 | 32 | 99.00 | 0.77 |

### Qwen3-32B

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 1024 / 128 | 32 | 768.82 | 6.01 |

### Qwen3-Coder-Next (FP8)

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 1024 / 128 | 32 | 50.43 | 0.39 |
| `g6e.12xlarge` | 4 | 1024 / 128 | 32 | 63.13 | 0.49 |

### Qwen3.5-27B (FP8)

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 1024 / 128 | 64 | 78.91 | 0.62 |
| `g6e.12xlarge` | 4 | 1024 / 128 | 64 | 88.58 | 0.69 |

> **Note — FP8 on A100:** A100 GPUs lack native FP8 support (requires compute capability 8.9+). vLLM dequantizes to BF16 at load, doubling weight
> memory. The `p4d.24xlarge` result also uses `--enforce-eager`.

### Qwen3.5-35B-A3B (FP8)

| Instance Type | TP | Input/Output Len | Prompts | Output tok/s | Requests/s |
| --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 1024 / 128 | 64 | 101.18 | 0.79 |
| `g6e.12xlarge` | 4 | 1024 / 128 | 64 | 115.24 | 0.90 |

## Latency by Model

### Qwen3.5-9B

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `g6.xlarge` | 1 | 4 | 18.20 | 18.22 | 18.24 | 18.24 |

### GPT-OSS-20B

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `g6e.xlarge` | 1 | 4 | 1.29 | 1.28 | 1.35 | 1.37 |

### Llama 3.3 70B Instruct

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 2 | 4.41 | 4.41 | 4.41 | 4.41 |
| `g6e.12xlarge` | 4 | 2 | 8.28 | 8.30 | 8.30 | 8.30 |

### Qwen3-32B

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 2 | 4.44 | 4.44 | 4.44 | 4.44 |

### Qwen3-Coder-Next (FP8)

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 2 | 2.13 | 2.13 | 2.14 | 2.14 |
| `g6e.12xlarge` | 4 | 2 | 2.32 | 2.33 | 2.34 | 2.35 |

### Qwen3.5-27B (FP8)

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 4 | 14.69 | 14.69 | 14.72 | 14.76 |
| `g6e.12xlarge` | 4 | 4 | 3.60 | 3.60 | 3.60 | 3.60 |

### Qwen3.5-35B-A3B (FP8)

| Instance Type | TP | Batch Size | Avg (s) | p50 (s) | p90 (s) | p99 (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `p4d.24xlarge` | 4 | 4 | 1.04 | 1.04 | 1.05 | 1.05 |
| `g6e.12xlarge` | 4 | 4 | 1.28 | 1.27 | 1.28 | 1.28 |

## Glossary

| Metric | Definition |
| --- | --- |
| **Output tok/s** | Output tokens generated per second across all concurrent requests |
| **TTFT** | Time to first token — latency from request submission to first token generated |
| **TPOT** | Time per output token — average inter-token latency after the first token |
| **TP** | Tensor parallelism degree (number of GPUs) |

## Running Your Own Benchmarks

The benchmarks above were produced using the
[vllm_benchmark_test.sh](https://github.com/aws/deep-learning-containers/blob/main/scripts/vllm/benchmark/vllm_benchmark_test.sh) script included in
this repository. It runs `vllm bench throughput` (offline, saturated) and `vllm bench latency` (fixed batch size) against a local model directory.

> **Warning — Input length override:** `vllm bench throughput` defaults to the `random` dataset when no `--dataset-path` is provided. The `random`
> dataset uses `--random-input-len` (default 1024) and ignores `--input-len`. Verify the actual input length in the benchmark log output.

To run a quick benchmark yourself:

```bash
# Throughput (offline, saturated)
docker run --gpus all \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  vllm bench throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 64

# Latency (fixed batch)
docker run --gpus all \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 1024 \
    --output-len 128 \
    --batch-size 4 \
    --num-iters 10
```

> **Note:** Results are specific to the {{ dlc_short }} image and may differ from upstream vLLM benchmarks due to curated patches and dependency
> versions.
