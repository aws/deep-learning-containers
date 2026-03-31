# Configuration

Configure the vLLM {{ dlc_short }} images using environment variables and server arguments.

## {{ sagemaker }} Environment Variables

When running on {{ sagemaker }}, the entrypoint converts any `SM_VLLM_*` environment variable into a vLLM server argument by stripping the prefix,
lowercasing, and replacing underscores with hyphens. For example, `SM_VLLM_MAX_MODEL_LEN=4096` becomes `--max-model-len 4096`.

| Variable | Description | Required |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID from Hugging Face Hub or S3 path | Yes |
| `HF_TOKEN` | Hugging Face token for gated models | For gated models |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | No |
| `SM_VLLM_MAX_MODEL_LEN` | Maximum sequence length | No |
| `SM_VLLM_ENFORCE_EAGER` | Set to `true` to disable CUDA graphs | No |

### Example

```python
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "HF_TOKEN": "<your_hf_token>",
    "SM_VLLM_MAX_MODEL_LEN": "4096",
    "SM_VLLM_ENFORCE_EAGER": "true",
}
```

## {{ ec2_short }} Server Arguments

When running on {{ ec2_short }}, {{ ecs_short }}, or {{ eks_short }}, pass arguments directly to the vLLM server:

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192
```

### Common Server Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model` | Model ID or path | Required |
| `--host` | Bind address | `localhost` |
| `--port` | Server port | `8000` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `1` |
| `--pipeline-parallel-size` | Number of pipeline parallel stages | `1` |
| `--max-model-len` | Maximum sequence length | Model default |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.9` |
| `--enforce-eager` | Disable CUDA graph for debugging | `false` |
| `--quantization` | Quantization method (awq, gptq, fp8, etc.) | None |
| `--dtype` | Model data type (auto, float16, bfloat16) | `auto` |
| `--enable-lora` | Enable LoRA adapter serving | `false` |

For the full list of arguments, see the [vLLM engine arguments documentation](https://docs.vllm.ai/en/latest/configuration/engine_args.html).

## Environment Variables

These environment variables work on both {{ ec2_short }} and {{ sagemaker }} platforms:

| Variable | Description |
| --- | --- |
| `HF_TOKEN` | Hugging Face authentication token |
| `HF_HOME` | Hugging Face cache directory |
| `VLLM_ATTENTION_BACKEND` | Override attention backend (e.g., `FLASH_ATTN`, `XFORMERS`) |
| `VLLM_USE_V1` | Enable vLLM V1 engine (`1` to enable) |
| `NCCL_DEBUG` | NCCL debug level (`INFO`, `WARN`) |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs |

For the full list, see the [vLLM environment variables documentation](https://docs.vllm.ai/en/latest/configuration/env_vars.html).

## Memory and Performance Tuning

### GPU Memory

```bash
# Use 95% of GPU memory (default is 90%)
--gpu-memory-utilization 0.95

# Limit context length to reduce memory
--max-model-len 4096
```

### Multi-GPU

```bash
# 4-way tensor parallelism
--tensor-parallel-size 4

# 2-way pipeline parallelism across nodes
--pipeline-parallel-size 2
```

### Throughput

```bash
# Increase max concurrent sequences
--max-num-seqs 512

# Enable chunked prefill for better scheduling
--enable-chunked-prefill
```
