# Configuration

## {{ sagemaker }} Environment Variables

When running on {{ sagemaker }}, the entrypoint converts any `SM_SGLANG_*` environment variable into an SGLang server argument by stripping the
prefix, lowercasing, and replacing underscores with hyphens. For example, `SM_SGLANG_TENSOR_PARALLEL_SIZE=4` becomes `--tensor-parallel-size 4`.

Boolean handling: `SM_SGLANG_TRUST_REMOTE_CODE=true` becomes `--trust-remote-code` (flag only). `=false` omits the flag entirely.

| Variable | Description | Required |
| --- | --- | --- |
| `SM_SGLANG_MODEL_PATH` | Model ID from Hugging Face Hub, S3 path, or local path | Yes |
| `HF_TOKEN` | Hugging Face token for gated models | For gated models |
| `SM_SGLANG_TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | No |
| `SM_SGLANG_DTYPE` | Model data type (auto, float16, bfloat16) | No |
| `SM_SGLANG_CONTEXT_LENGTH` | Context length for the model | No |
| `SM_SGLANG_REASONING_PARSER` | Reasoning parser (e.g., `qwen3`) | No |
| `SM_SGLANG_HOST` | Server bind address (default: `0.0.0.0`) | No |
| `SM_SGLANG_PORT` | Server port (default: `8080`) | No |

### Example

```python
env = {
    "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-8B-Instruct",
    "HF_TOKEN": "<your_hf_token>",
    "SM_SGLANG_TENSOR_PARALLEL_SIZE": "4",
    "SM_SGLANG_CONTEXT_LENGTH": "8192",
    "SM_SGLANG_TRUST_REMOTE_CODE": "true",
}
```

> **Note:** The SageMaker entrypoint defaults to `--port 8080` and `--host 0.0.0.0`. The model path defaults to `/opt/ml/model` if
> `SM_SGLANG_MODEL_PATH` is not set.

## {{ ec2_short }} Server Arguments

When running on {{ ec2_short }}, {{ ecs_short }}, or {{ eks_short }}, pass arguments directly to `python3 -m sglang.launch_server`:

```bash
docker run --gpus all -p 30000:30000 <image_uri> \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 4 \
  --context-length 8192
```

### Common Server Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Model ID or path | Required |
| `--host` | Bind address | `127.0.0.1` |
| `--port` | Server port | `30000` |
| `--tp` | Number of GPUs for tensor parallelism | `1` |
| `--context-length` | Maximum context length | Model default |
| `--dtype` | Model data type (auto, float16, bfloat16) | `auto` |
| `--mem-fraction-static` | Fraction of GPU memory for static allocation | `0.88` |
| `--trust-remote-code` | Trust remote code from Hugging Face | `false` |
| `--reasoning-parser` | Reasoning parser for thinking models | None |

For the full list of arguments, see the [SGLang documentation](https://docs.sglang.ai/).

## Environment Variables

These environment variables work on both {{ ec2_short }} and {{ sagemaker }} platforms:

| Variable | Description |
| --- | --- |
| `HF_TOKEN` | Hugging Face authentication token |
| `HF_HOME` | Hugging Face cache directory |
| `NCCL_DEBUG` | NCCL debug level (`INFO`, `WARN`) |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs |

## Memory and Performance Tuning

### GPU Memory

```bash
# Use 92% of GPU memory for static allocation (default is 88%)
--mem-fraction-static 0.92

# Limit context length to reduce memory
--context-length 4096
```

### Multi-GPU

```bash
# 4-way tensor parallelism
--tp 4

# 2-way data parallelism
--dp 2
```

### Throughput

```bash
# Enable chunked prefill for better scheduling
--chunked-prefill-size 8192

# Enable torch.compile for optimized execution
--enable-torch-compile
```

## See Also

- [Deployment](deployment.md) — production deployment examples
- [Benchmarks](benchmarks.md) — performance tuning results
- [Supported Models](supported_models.md) — tested models and compatibility
