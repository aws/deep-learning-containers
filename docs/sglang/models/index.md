# Supported Models

All models listed below are regression-tested on every DLC SGLang release and work with the images listed on the [Overview](../index.md) page.

The **Coverage** column indicates test depth: *Smoke* runs on every PR; *Benchmark* runs throughput and latency tests with pass/fail thresholds before
release. A *Smoke + Benchmark* tag means both apply.

## Tested Models

| Family | Model | Coverage |
| --- | --- | --- |
| **Llama** | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Benchmark |
| **Qwen** | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Benchmark |
|  | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Smoke + Benchmark |
|  | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Benchmark |
|  | [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) | Benchmark |
|  | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) | Benchmark |
|  | [Qwen/Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) | Benchmark |
| **GPT-OSS** | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Benchmark |
| **DeepSeek** | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai) | Benchmark |

## Custom Models

Any model supported by upstream SGLang should work. To serve a model not listed above:

```bash
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-cuda \
  --model-path <org>/<model-name>
```

Models can also be loaded from a local path (`-v /path:/model --model-path /model`). See the
[SGLang supported models list](https://docs.sglang.ai/supported_models/generative_models.html) for the full upstream coverage.
