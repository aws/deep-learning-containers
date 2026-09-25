# EC2 Deployment

The container runs the vLLM OpenAI-compatible API server on port 8000. Any `vllm serve` flag may be appended to `docker run`. See
[Configuration](../configuration.md) for the full list of server arguments.

## Single GPU

```bash
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000
```

For gated models (Llama, Gemma, etc.), pass `-e HF_TOKEN=<your_hf_token>`.

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256
  }'
```

## Multi-GPU (Tensor Parallelism)

For models that require multiple GPUs (e.g., 70B+):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port 8000
```

`--ipc=host` enables shared memory between GPU processes.

## Loading Models from S3

The image bundles [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — pass an `s3://` URI as the model and add
`--load-format runai_streamer` to stream weights directly from S3 to GPU memory:

```bash
docker run --gpus all -p 8000:8000 \
  -e AWS_REGION=us-west-2 \
  -e AWS_ACCESS_KEY_ID=<key> \
  -e AWS_SECRET_ACCESS_KEY=<secret> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model s3://<bucket>/<prefix>/ \
  --load-format runai_streamer \
  --host 0.0.0.0 --port 8000
```

The S3 prefix should contain a Hugging Face model layout (`config.json`, `tokenizer.json`, and one or more `*.safetensors` files). On EC2 with an
attached instance role, the AWS credentials may be omitted — the container will pick them up from IMDS. See the
[Run:ai streamer docs](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html) for sharded loading and tuning options.

## Graviton (ARM64 CPU)

On AWS Graviton instances (for example `c7g`, `c8g`, `m8g`, `r8g`), use the CPU image. No `--gpus` flag is needed:

```bash
docker run -p 8000:8000 --shm-size=4g \
  public.ecr.aws/deep-learning-containers/vllm:server-arm64-cpu \
  --model Qwen/Qwen3.5-2B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --host 0.0.0.0 --port 8000
```

The CPU image sets these defaults at startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VLLM_CPU_KVCACHE_SPACE` | 40% of host RAM (minimum 2 GiB) | KV-cache size in GiB. GPU memory flags such as `--gpu-memory-utilization` have no effect on CPU. |
| `VLLM_CPU_OMP_THREADS_BIND` | `nobind` | OpenMP thread binding. Set to `auto` or a core list (for example `0-31`) to pin threads. |
| `LD_PRELOAD` | `libtcmalloc_minimal.so.4` | tcmalloc for lower allocator overhead |

Override any of them with `-e`, for example `-e VLLM_CPU_KVCACHE_SPACE=16`. Use `bfloat16` weights; GGUF models are not supported on the CPU backend.
See [vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html) for more tuning options.

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see [recipes.vllm.ai](https://recipes.vllm.ai/).
