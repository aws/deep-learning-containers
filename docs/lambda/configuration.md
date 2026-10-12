# Configuration

The images take no command-line arguments — {{ lambda }} controls `ENTRYPOINT` and `CMD`. Configure them with environment variables, set either in
the Dockerfile with `ENV` or on the function. Function-level variables win, so one image can serve several models.

## vLLM (`vllm:cuda`)

Read by the default handler and passed to `vllm serve`.

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Positional model: Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the default `model` field in requests |
| `VLLM_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` to stream weights into GPU memory |
| `VLLM_GPU_MEM_UTIL` | `0.8` | `--gpu-memory-utilization` |
| `VLLM_MAX_MODEL_LEN` | Unset (model default) | `--max-model-len` |
| `VLLM_TP_SIZE` | Visible GPU count | `--tensor-parallel-size` |
| `VLLM_SERVER_PORT` | `8000` | `--port` (loopback only) |
| `VLLM_SERVER_TIMEOUT` | `600` | Seconds to wait for the server's health endpoint, and the per-request timeout |

## SGLang (`sglang:cuda`)

Read by the default handler and passed to `python -m sglang.launch_server`.

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | `--model-path`: Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the default `model` field in requests |
| `SGLANG_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` |
| `SGLANG_MEM_FRACTION` | `0.8` | `--mem-fraction-static` |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | `--attention-backend` |
| `SGLANG_MAX_TOTAL_TOKENS` | Unset | `--max-total-tokens` |
| `SGLANG_TP_SIZE` | Visible GPU count | `--tp-size` |
| `SGLANG_SERVER_PORT` | `8000` | `--port` (loopback only) |
| `SGLANG_SERVER_TIMEOUT` | `600` | Seconds to wait for the server's health endpoint, and the per-request timeout |

`MODEL_ID` has a default so the image serves out of the box, but it is a 0.5B smoke-test model. Always set it.

Tensor parallelism defaults to the number of visible GPUs, which is 1 — {{ lambda_short }} assigns one GPU per execution environment. Multi-GPU and
multi-node serving are not supported; leave `*_TP_SIZE` alone unless an execution environment is granted more than one GPU.

To pass a server flag these variables do not cover, replace the default handler. See
[Write your own handler](deployment/lambda.md#write-your-own-handler).

## Baked-in paths

`/tmp` is the only writable directory, so the images already redirect everything that writes:

| Variable | Value | Images |
| --- | --- | --- |
| `HOME` | `/tmp` | All |
| `XDG_CONFIG_HOME` | `/tmp/.config` | All |
| `XDG_CACHE_HOME` | `/tmp/.cache` | All |
| `TORCH_HOME` | `/tmp/torch` | PyTorch, vLLM, SGLang |
| `HF_HOME` | `/tmp/huggingface` | PyTorch, vLLM, SGLang |
| `CUPY_CACHE_DIR` | `/tmp/cupy_cache` | CuPy |
| `CUDA_HOME` | `/usr/local/cuda` | CuPy, vLLM, SGLang |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` on PyTorch, `compute,utility` elsewhere | All |

Anything downloaded into these paths counts against `EphemeralStorage`. Size it for the model when you are not streaming from S3.

## Function settings that matter

Set on the function, not the image. For the full list, see the
[{{ lambda_short }} Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html).

| Setting | Notes |
| --- | --- |
| `AcceleratorMemorySize` | GPU memory, in GB: `3`, `6`, `12`, `16`, `24`, `48`. A minimum — {{ lambda_short }} picks the smallest qualifying GPU |
| `PerExecutionEnvironmentMaxConcurrency` | Default 4, maximum 48. Leave at the default for vLLM and SGLang; the server batches requests itself |
| `ExecutionEnvironmentMemoryGiBPerVCpu` | Default 2. Raise it when `MemorySize` would otherwise claim every vCPU on the instance |
| `EphemeralStorage` | 512 MB to 32 GB. Only needs to hold the model if you are not streaming it |
| `Timeout` | Also bounds initialization, at the greater of 130 s and this value, capped at 900 s |

## Full reference

- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [Run:ai Model Streamer in vLLM](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)
