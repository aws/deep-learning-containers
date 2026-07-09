# Supported Models

All models listed below are regression-tested on every DLC vLLM release and work with the images listed on the [Overview](../index.md) page.

The **Coverage** column indicates test depth: *Smoke* runs on every PR; *Benchmark* runs throughput and latency tests with pass/fail thresholds before
release. A *Smoke + Benchmark* tag means both apply.

## Tested Models

| Family | Model | Coverage |
| --- | --- | --- |
| **Llama** | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Benchmark |
| **Qwen** | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Benchmark |
|  | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Smoke + Benchmark |
|  | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Benchmark |
|  | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Benchmark |
|  | [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) | Benchmark |
|  | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Benchmark |
|  | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Benchmark |
|  | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Benchmark |
|  | [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) | Benchmark |
|  | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Smoke + Benchmark |
|  | [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Smoke + Benchmark |
|  | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Benchmark |
| **Gemma** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | Benchmark |
|  | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | Benchmark |
|  | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Benchmark |
| **GPT-OSS** | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Benchmark |
| **JetBrains** | [JetBrains/Mellum2-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) | Smoke + Benchmark |

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see [recipes.vllm.ai](https://recipes.vllm.ai/).

## Custom Models

Any model supported by upstream vLLM should work. To serve a model not listed above:

```bash
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model <org>/<model-name>
```

Models can also be loaded from a local path (`-v /path:/model --model /model`) or streamed from S3 — see
[Loading Models from S3](../deployment/ec2.md#loading-models-from-s3).
