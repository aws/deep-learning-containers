# Configuration

## EC2 / EKS (`server-cuda`)

Pass SGLang server arguments directly to `docker run`. The entrypoint forwards them to `python3 -m sglang.launch_server`:

```bash
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-cuda \
  --model-path openai/gpt-oss-20b \
  --tp 4 \
  --context-length 4096
```

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Model ID or path (required) | — |
| `--host` | Bind address | `127.0.0.1` |
| `--port` | Server port | `30000` |
| `--tp` | Tensor-parallel size (number of GPUs) | `1` |
| `--context-length` | Maximum sequence length | Model default |
| `--mem-fraction-static` | Fraction of GPU memory for the KV cache pool | Auto |
| `--dtype` | Data type (auto, bfloat16, float16) | `auto` |
| `--quantization` | Quantization method (fp8, awq, gptq, …) | None |
| `--trust-remote-code` | Allow custom model code from the Hub | `false` |
| `--disable-piecewise-cuda-graph` | Disable experimental piecewise CUDA graph capture | `false` |

For gated models (Llama, Gemma, etc.), pass `-e HF_TOKEN=<your_hf_token>`.

## Amazon SageMaker AI (`server-sagemaker-cuda`)

The SageMaker image serves on **port 8080** and accepts SGLang flags via `SM_SGLANG_*` environment variables. Each variable is converted to the
corresponding SGLang flag — the name is lowercased and underscores become hyphens (e.g., `SM_SGLANG_CONTEXT_LENGTH=4096` → `--context-length 4096`).
Boolean values follow shell convention: `true` becomes a bare flag (`SM_SGLANG_TRUST_REMOTE_CODE=true` → `--trust-remote-code`), and `false` omits the
flag entirely.

| Variable | Description | Default |
| --- | --- | --- |
| `SM_SGLANG_MODEL_PATH` | Model ID or path (defaults to `/opt/ml/model` when SageMaker mounts artifacts) | `/opt/ml/model` |
| `SM_SGLANG_TP` | Tensor-parallel size (number of GPUs) | `1` |
| `SM_SGLANG_CONTEXT_LENGTH` | Maximum sequence length | Model default |
| `SM_SGLANG_MEM_FRACTION_STATIC` | Fraction of GPU memory for the KV cache pool | Auto |
| `SM_SGLANG_DTYPE` | Data type (auto, bfloat16, float16) | `auto` |
| `SM_SGLANG_QUANTIZATION` | Quantization method (fp8, awq, gptq, …) | None |
| `SM_SGLANG_TRUST_REMOTE_CODE` | Allow custom model code from the Hub | `false` |
| `SM_SGLANG_PORT` | Server port | `8080` |
| `SM_SGLANG_HOST` | Bind address | `0.0.0.0` |
| `HF_TOKEN` | Hugging Face token for gated models | — |

The entrypoint always supplies `--port 8080`, `--host 0.0.0.0`, and `--model-path /opt/ml/model` unless you override them via the matching
`SM_SGLANG_*` variable.

## Full Reference

- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [SGLang hyperparameter tuning](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)
