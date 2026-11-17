# Configuration

{{ lambda }} controls `ENTRYPOINT` and `CMD`, so you configure these images with environment variables rather than command-line arguments. Set them in
the Dockerfile with `ENV`, or on the function. Variables set on the function take precedence, so one image can serve several models.

## vLLM (`vllm:cuda`)

Read by the [engine extension](deployment/lambda.md#the-engine-extension) and passed to `vllm serve`.

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the request's default `model` |
| `VLLM_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` |
| `VLLM_GPU_MEM_UTIL` | `0.8` | `--gpu-memory-utilization` |
| `VLLM_MAX_MODEL_LEN` | Unset (model default) | `--max-model-len` |
| `VLLM_TP_SIZE` | Visible GPU count | `--tensor-parallel-size` |
| `VLLM_SERVER_ARGS` | Unset | Any further `vllm serve` flags, shell-quoted |
| `VLLM_SERVER_PORT` | `8000` | `--port`, loopback only |
| `VLLM_SERVER_TIMEOUT` | `600` | Seconds to wait for the engine to be ready |

## SGLang (`sglang:cuda`)

Read by the [engine extension](deployment/lambda.md#the-engine-extension) and passed to `python -m sglang.launch_server`.

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | `--model-path`. Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the request's default `model` |
| `SGLANG_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` |
| `SGLANG_MEM_FRACTION` | `0.8` | `--mem-fraction-static` |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | `--attention-backend` |
| `SGLANG_MAX_TOTAL_TOKENS` | Unset | `--max-total-tokens` |
| `SGLANG_TP_SIZE` | Visible GPU count | `--tp-size` |
| `SGLANG_SERVER_ARGS` | Unset | Any further `sglang.launch_server` flags, shell-quoted |
| `SGLANG_SERVER_PORT` | `8000` | `--port`, loopback only |
| `SGLANG_SERVER_TIMEOUT` | `600` | Seconds to wait for the engine to be ready |

`MODEL_ID` has a default so that the image serves without configuration, but it points at a small test model. Set it for any real workload.

Tensor parallelism defaults to the number of visible GPUs, which is 1, because {{ lambda_short }} assigns one GPU to each execution environment.
Multi-GPU and multi-node serving are not supported, so leave `*_SERVER_ARGS` and `*_TP_SIZE` at their defaults unless an execution environment is
granted more than one GPU.

The handler takes its per-request timeout from the invocation deadline, so set the function's `Timeout` rather than a variable.

## Engine extension (`vllm:cuda`, `sglang:cuda`)

| Variable | Default | Effect |
| --- | --- | --- |
| `ENGINE_EXTENSION` | `on` | Set to `off` to run your own server |
| `ENGINE_HEALTH_INTERVAL` | `10` | Seconds between probes once serving |
| `ENGINE_HEALTH_TIMEOUT` | `10` vLLM, `30` SGLang | Seconds allowed for each probe |
| `ENGINE_HEALTH_FAILURES` | `3` | Failed probes before the environment is replaced |

SGLang's health check can run a one-token generation, so it uses the longer probe timeout. For the `off` path, see
[Running your own server instead](deployment/lambda.md#running-your-own-server-instead).

## Baked-in paths

`/tmp` is the only writable directory, so the images redirect everything that writes.

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

Anything downloaded into these paths counts toward `EphemeralStorage`. Size it for the model when you are not streaming from S3.

## Function settings that matter

Set these on the function rather than in the image. For the full list, see the
[{{ lambda_short }} Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html).

| Setting | Notes |
| --- | --- |
| `AcceleratorMemorySize` | GPU memory in GB: `3`, `6`, `12`, `16`, `24`, `48`. A minimum, not an exact size |
| `PerExecutionEnvironmentMaxConcurrency` | Default 4, maximum 48. Leave at the default for vLLM and SGLang |
| `ExecutionEnvironmentMemoryGiBPerVCpu` | Default 2. Raise it to claim fewer vCPUs per environment |
| `EphemeralStorage` | 512 MB to 32 GB. Only holds the model if you are not streaming it |
| `Timeout` | Maximum duration of one invocation. At least 900 for vLLM and SGLang |

`AcceleratorMemorySize` is a minimum, so {{ lambda_short }} selects the smallest GPU that meets it. Leave `PerExecutionEnvironmentMaxConcurrency` at
its default on the engine images, because the server batches requests itself. Raise `ExecutionEnvironmentMemoryGiBPerVCpu` when `MemorySize` would
otherwise claim every vCPU on the instance. Keep `VLLM_SERVER_TIMEOUT` or `SGLANG_SERVER_TIMEOUT` at or below `Timeout`, and just under 900 for large
models.

## Full reference

- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [Run:ai Model Streamer in vLLM](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)
