# Configuration

{{ lambda }} controls `ENTRYPOINT` and `CMD`, so you configure these images with environment variables rather than command-line arguments. Set them in
the Dockerfile with `ENV`, or on the function. Variables set on the function take precedence, so one image can serve several models.

To switch models, update `MODEL_ID` on the function and publish a new version. No rebuild is needed.
[UpdateFunctionConfiguration](https://docs.aws.amazon.com/lambda/latest/api/API_UpdateFunctionConfiguration.html) replaces the whole variable map
rather than merging into it, so pass the variables you already set, such as `VLLM_LOAD_FORMAT`, along with the new `MODEL_ID`.

## vLLM (`vllm:cuda`)

Read by the [engine extension](deployment/lambda.md#the-engine-extension) and passed to `vllm serve`.

<div class="wrap-table" markdown>

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | Positional model: Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the default `model` field in requests |
| `VLLM_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` to stream weights into GPU memory |
| `VLLM_GPU_MEM_UTIL` | `0.8` | `--gpu-memory-utilization` |
| `VLLM_MAX_MODEL_LEN` | Unset (model default) | `--max-model-len` |
| `VLLM_TP_SIZE` | Visible GPU count | `--tensor-parallel-size` |
| `VLLM_SERVER_ARGS` | Unset | Any further `vllm serve` flags, shell-quoted |
| `VLLM_SERVER_PORT` | `8000` | `--port` (loopback only); also read by the handler |
| `VLLM_SERVER_TIMEOUT` | `600` | Seconds to wait for the engine to be ready. Raise to just under the function `Timeout` for large models |
| `HF_TOKEN` | Unset | Hugging Face token. Raises download rate limits and reaches private or gated models |

</div>

## SGLang (`sglang:cuda`)

Read by the [engine extension](deployment/lambda.md#the-engine-extension) and passed to `python -m sglang.launch_server`.

<div class="wrap-table" markdown>

| Variable | Default | Maps to |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen2.5-0.5B-Instruct` | `--model-path`: Hugging Face ID, local path, or `s3://` prefix |
| `SERVED_MODEL_NAME` | Unset | `--served-model-name`, and the default `model` field in requests |
| `SGLANG_LOAD_FORMAT` | Unset | `--load-format`. Set `runai_streamer` with an `s3://` `MODEL_ID` |
| `SGLANG_MEM_FRACTION` | `0.8` | `--mem-fraction-static` |
| `SGLANG_ATTENTION_BACKEND` | `flashinfer` | `--attention-backend` |
| `SGLANG_MAX_TOTAL_TOKENS` | Unset | `--max-total-tokens` |
| `SGLANG_TP_SIZE` | Visible GPU count | `--tp-size` |
| `SGLANG_SERVER_ARGS` | Unset | Any further `sglang.launch_server` flags, shell-quoted |
| `SGLANG_SERVER_PORT` | `8000` | `--port` (loopback only); also read by the handler |
| `SGLANG_SERVER_TIMEOUT` | `600` | Seconds to wait for the engine to be ready. Raise to just under the function `Timeout` for large models |
| `HF_TOKEN` | Unset | Hugging Face token. Raises download rate limits and reaches private or gated models |

</div>

`MODEL_ID` has a default so that the image serves without configuration, but it points at a small test model. Set it for any real workload.

Tensor parallelism defaults to the number of visible GPUs, which is 1, because {{ lambda_short }} assigns one GPU to each execution environment.
Multi-GPU and multi-node serving are not supported, so leave `*_SERVER_ARGS` and `*_TP_SIZE` at their defaults unless an execution environment is
granted more than one GPU.

The handler takes its per-request timeout from the invocation deadline, so set the function's `Timeout` rather than a variable.

## Engine extension (`vllm:cuda`, `sglang:cuda`)

<div class="wrap-table" markdown>

| Variable | Default | Effect |
| --- | --- | --- |
| `ENGINE_EXTENSION` | `on` | Set to `off` for functions that [run their own server](deployment/lambda.md#running-your-own-server-instead). The extension registers without starting an engine |
| `ENGINE_HEALTH_INTERVAL` | `10` | Seconds between `/health` probes once serving |
| `ENGINE_HEALTH_TIMEOUT` | `10` vLLM, `30` SGLang | Seconds allowed for each probe. SGLang's health check can run a one-token generation, so it uses the longer default |
| `ENGINE_HEALTH_FAILURES` | `3` | Consecutive failed probes before the environment is recycled |

</div>

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

At runtime `/tmp` starts empty, and the rest of the image, including `/opt` and `/var/task`, is read-only. Anything your Dockerfile writes under
`/tmp` at build time is therefore gone when the function starts, while files elsewhere in the image stay and can be read. Library downloads land under
`/tmp` by default because of the variables above: a `RUN` step that loads `torchvision` weights with `weights=DEFAULT` saves them under `TORCH_HOME`,
and every new execution environment downloads them again at initialization, which fails without internet access. To bake weights into the image, save
them outside `/tmp` and load them by path:

```dockerfile
RUN python3 -c "import torch; from torchvision import models; \
torch.save(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).state_dict(), '/var/task/resnet18.pth')" \
  && rm -rf /tmp/torch
```

```python
model = models.resnet18()
model.load_state_dict(torch.load("/var/task/resnet18.pth", map_location="cpu", weights_only=True))
```

For Hugging Face models, download them at build time into a directory outside `/tmp`, for example with
`huggingface-cli download <model> --local-dir /opt/model` or `snapshot_download(<model>, local_dir="/opt/model")`, and pass `/opt/model` as the model
instead of the model ID. Keep `TORCH_HOME` and `HF_HOME` themselves on `/tmp`: libraries write their caches there at runtime, and only `/tmp` is
writable.

## Function settings that matter

Set these on the function rather than in the image. For the full list, see the
[GPU-accelerated Event Functions](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-gpu.html) in the {{ lambda_short }} Developer
Guide.

<div class="wrap-table" markdown>

| Setting | Notes |
| --- | --- |
| `AcceleratorMemorySize` | GPU memory, in GB: `3`, `6`, `12`, `16`, `24`, `48`. This is a minimum, and {{ lambda_short }} selects the smallest GPU that meets it |
| `PerExecutionEnvironmentMaxConcurrency` | Default 4, maximum 48. Leave this at the default for vLLM and SGLang, because the server batches requests itself |
| `ExecutionEnvironmentMemoryGiBPerVCpu` | Default 2. Raise it when `MemorySize` would otherwise claim every vCPU on the instance |
| `EphemeralStorage` | 512 MB to 32 GB. This only needs to hold the model if you are not streaming it from S3 |
| `Timeout` | Maximum duration of one invocation. Raise it for longer invokes, and to 900 for vLLM and SGLang to give the model time to load at startup. Keep `VLLM_SERVER_TIMEOUT` / `SGLANG_SERVER_TIMEOUT` at or below it |

</div>

## Full reference

- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)
- [Run:ai Model Streamer in vLLM](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/)
