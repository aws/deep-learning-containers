# EC2 Deployment

The container runs the upstream `llama-server` on port 8080. On {{ ec2_short }} you supply the model and any tuning as **`llama-server` arguments**
appended to `docker run` — the entrypoint forwards them straight through. See [Configuration](../configuration.md) for the options the DLC adds on top.

The server is **unauthenticated by default** and binds `0.0.0.0`. Run it inside a private network (security group / VPC), and set `LLAMA_API_KEY` to
require a bearer token — see [Authentication](#authentication).

## CPU (x86)

Fetch a quantized GGUF from HuggingFace at startup with `--hf-repo` / `--hf-file`:

```bash
docker run -d -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/llama-cpp:server-cpu-v1 \
  --hf-repo ggml-org/Qwen2.5-0.5B-Instruct-GGUF \
  --hf-file qwen2.5-0.5b-instruct-q4_0.gguf \
  --ctx-size 4096
```

`llama-server` binds the socket only after the model has loaded, so `/health` refuses connections until the model is resident. Wait for readiness,
then call the OpenAI-compatible API:

```bash
until curl -sf http://localhost:8080/health > /dev/null; do sleep 5; done

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "In one sentence, what is llama.cpp?"}]
  }'
```

To serve a **local GGUF** instead, mount it and point `--model` at the mount:

```bash
docker run -d -p 8080:8080 \
  -v /path/to/models:/models:ro \
  public.ecr.aws/deep-learning-containers/llama-cpp:server-cpu-v1 \
  --model /models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --ctx-size 4096
```

## GPU (x86, CUDA)

The GPU image is identical except it offloads layers to the NVIDIA GPU with `--n-gpu-layers` (`999` offloads all layers). Run with `--gpus all`:

```bash
docker run -d --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/llama-cpp:server-cuda-v1 \
  --hf-repo ggml-org/Qwen2.5-0.5B-Instruct-GGUF \
  --hf-file qwen2.5-0.5b-instruct-q4_0.gguf \
  --n-gpu-layers 999 \
  --ctx-size 4096
```

The entrypoint activates CUDA forward-compatibility automatically when the host NVIDIA driver is older than the CUDA 13.0.2 runtime requires — no extra
flag needed. If the container starts without a visible GPU it falls back to the CPU backend.

## Graviton (ARM64)

The Graviton image runs the same way on an ARM64 (`c7g`, `c8g`, `r8g`, …) instance — there is no GPU offload, so omit `--n-gpu-layers`:

```bash
docker run -d -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/llama-cpp-arm64:server-cpu-v1 \
  --hf-repo ggml-org/Qwen2.5-0.5B-Instruct-GGUF \
  --hf-file qwen2.5-0.5b-instruct-q4_0.gguf \
  --ctx-size 4096
```

## Authentication

By default the endpoint accepts unauthenticated requests. Set `LLAMA_API_KEY` to require a bearer token on every request:

```bash
docker run -d -p 8080:8080 \
  -e LLAMA_API_KEY=my-secret-key \
  public.ecr.aws/deep-learning-containers/llama-cpp:server-cpu-v1 \
  --hf-repo ggml-org/Qwen2.5-0.5B-Instruct-GGUF \
  --hf-file qwen2.5-0.5b-instruct-q4_0.gguf
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Tuning and the CLI

- All `llama-server` flags (`--ctx-size`, `--parallel`, `--threads`, `--batch-size`, `--n-gpu-layers`, …) are passed as container arguments — see
  [Configuration](../configuration.md).
- The image also bundles `llama-cli` and `llama-bench`. Override the entrypoint to run them, e.g.
  `docker run --rm --entrypoint llama-bench <image> ...`.
