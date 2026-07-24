# EC2 Deployment

The container runs the SGLang OpenAI-compatible API server on port 30000. Any `sglang.launch_server` flag may be appended to `docker run`. See
[Configuration](../configuration.md) for the full list of server arguments.

## Single GPU

```bash
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-cuda \
  --model-path openai/gpt-oss-20b \
  --host 0.0.0.0 --port 30000
```

For gated models (Llama, Gemma, etc.), pass `-e HF_TOKEN=<your_hf_token>`.

Send a request:

```bash
curl http://localhost:30000/v1/chat/completions \
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
docker run --gpus all --ipc=host -p 30000:30000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/sglang:server-cuda \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --tp 8 \
  --host 0.0.0.0 --port 30000
```

`--ipc=host` enables shared memory between GPU processes.

## Vision Grounding (Multimodal)

The images support multimodal vision-grounding models such as [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B), which
return bounding boxes for objects matching a text description. These models ship a custom Hugging Face processor, so pass `--trust-remote-code`:

```bash
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-cuda \
  --model-path nvidia/LocateAnything-3B \
  --trust-remote-code \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
```

Send an image with a grounding prompt using the OpenAI chat format with an `image_url` content part:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/LocateAnything-3B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/car.jpg"}},
        {"type": "text", "text": "Locate all the instances that matches the following description: car"}
      ]
    }],
    "max_tokens": 2048,
    "skip_special_tokens": false
  }'
```

The response content contains `<box>x1, y1, x2, y2</box>` coordinates for each detected instance. Pass `"skip_special_tokens": false` so the box
tokens are preserved in the output.

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see the
[SGLang hyperparameter tuning guide](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html). A few notes from our regression suite:

- **FP8 MoE models** (e.g., Qwen3 Coder, Qwen3.5-35B-A3B): block-quantized weights require each tensor-parallel shard's gate/up `output_size` to be a
  multiple of 128. On an 8-GPU host, use `--tp 4` rather than `--tp 8`.
- **Large dense models** (e.g., Llama 3.3 70B): pass `--mem-fraction-static 0.88` and `--disable-piecewise-cuda-graph` to reduce compile-time host
  memory during warmup.
