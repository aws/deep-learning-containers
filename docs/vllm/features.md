# Features

Key vLLM features available in the {{ dlc_short }} images.

## Quantization

Serve quantized models to reduce GPU memory usage and increase throughput.

### FP8

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --quantization fp8
```

### AWQ

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model TheBloke/Llama-2-7B-Chat-AWQ \
  --quantization awq
```

### GPTQ

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model TheBloke/Llama-2-7B-Chat-GPTQ \
  --quantization gptq
```

For all supported quantization methods, see the [vLLM quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/).

## LoRA Adapters

Serve multiple LoRA adapters on a single base model:

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules my-adapter=s3://<bucket>/my-lora-adapter
```

Send a request specifying the adapter:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-adapter",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

## Structured Outputs

Force the model to produce valid JSON matching a schema:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Give me a user profile"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "user_profile",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

## Tool Calling

Enable function calling with compatible models:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What is the weather in Seattle?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

## Speculative Decoding

Accelerate generation using a smaller draft model:

```bash
docker run --gpus all -p 8000:8000 <image_uri> \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 5 \
  --tensor-parallel-size 8
```

## OpenAI-Compatible API

The vLLM {{ dlc_short }} images expose an OpenAI-compatible API server. Use any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain transformers briefly"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Supported Endpoints

| Endpoint | Description |
| --- | --- |
| `/v1/chat/completions` | Chat completions (streaming supported) |
| `/v1/completions` | Text completions |
| `/v1/models` | List loaded models |
| `/v1/embeddings` | Text embeddings (with embedding models) |
| `/health` | Health check |
