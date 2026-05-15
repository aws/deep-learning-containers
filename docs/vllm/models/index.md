# Supported Models

All models listed below are regression-tested on every DLC vLLM release and work with the images listed on the [Overview](../index.md) page.

## Tested Models

| Family | Model |
|---|---|
| **Llama** | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| **Qwen** | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| | [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) |
| | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| | [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) |
| | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| | [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Gemma** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
| **MiniMax** | [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |
| **GLM** | [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) |
| **GPT-OSS** | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see [recipes.vllm.ai](https://recipes.vllm.ai/).

## Custom Models

Any model supported by upstream vLLM should work. To serve a model not listed above:

```bash
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model <org>/<model-name>
```

Models can also be loaded from S3 (`--model s3://<bucket>/<prefix>/`) or a local path (`-v /path:/model --model /model`).
