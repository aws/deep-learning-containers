# Configuration

## EC2 / EKS (`server-cuda`)

Pass vLLM server arguments directly to `docker run`:

```bash
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model openai/gpt-oss-20b \
  --tensor-parallel-size 4 \
  --max-model-len 4096
```

| Argument | Description | Default |
| --- | --- | --- |
| `--model` | Model ID or path (required) | — |
| `--host` | Bind address | `localhost` |
| `--port` | Server port | `8000` |
| `--tensor-parallel-size` | Number of GPUs | `1` |
| `--max-model-len` | Maximum sequence length | Model default |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.9` |
| `--enforce-eager` | Disable CUDA graphs | `false` |
| `--quantization` | Quantization method (awq, gptq, fp8, …) | None |
| `--dtype` | Data type (auto, float16, bfloat16) | `auto` |

For gated models (Llama, Gemma, etc.), pass `-e HF_TOKEN=<your_hf_token>`.

## Amazon SageMaker AI (`server-sagemaker-cuda`)

The SageMaker image serves on **port 8080** and accepts vLLM flags via `SM_VLLM_*` environment variables. Each variable is converted to the
corresponding vLLM flag (e.g., `SM_VLLM_MAX_MODEL_LEN=4096` → `--max-model-len 4096`). Boolean values follow shell convention: `true` becomes a bare
flag (`SM_VLLM_ENFORCE_EAGER=true` → `--enforce-eager`), and `false` omits the flag entirely.

| Variable | Description | Default |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID or path (auto-detected from `/opt/ml/model` or `HF_MODEL_ID` if unset) | — |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs | `1` |
| `SM_VLLM_MAX_MODEL_LEN` | Maximum sequence length | Model default |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory to use | `0.9` |
| `SM_VLLM_ENFORCE_EAGER` | Disable CUDA graphs | `false` |
| `SM_VLLM_QUANTIZATION` | Quantization method (awq, gptq, fp8, …) | None |
| `SM_VLLM_DTYPE` | Data type (auto, float16, bfloat16) | `auto` |
| `HF_MODEL_ID` | Hugging Face model ID (fallback when `SM_VLLM_MODEL` is unset and `/opt/ml/model` is empty) | — |
| `HF_TOKEN` | Hugging Face token for gated models | — |

### Standard-Supervisor Settings

The SageMaker image includes [standard-supervisor](https://github.com/aws/model-hosting-container-standards) for process management and platform
integrations:

| Variable | Description | Default |
| --- | --- | --- |
| `PROCESS_AUTO_RECOVERY` | Auto-restart vLLM on crash | `true` |
| `PROCESS_MAX_START_RETRIES` | Max restart attempts before giving up | `3` |
| `STANDARD_AUTO_INSTALL_REQ` | Auto-install requirements.txt from model artifacts | `true` |
| `STANDARD_PIP_ARGS` | Custom pip arguments for dependency installation | — |

## Full Reference

- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [vLLM environment variables](https://docs.vllm.ai/en/latest/configuration/env_vars.html)
- [Standard-supervisor documentation](https://github.com/aws/model-hosting-container-standards)
