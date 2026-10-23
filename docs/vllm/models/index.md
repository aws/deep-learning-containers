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
| **Baidu** | [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | Smoke |

### Graviton CPU

The `vllm-arm64` images are tested on every release with:

| Model | Coverage |
| --- | --- |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Smoke |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | SageMaker endpoint (`ml.c7g`, `ml.c8g`) |

## Reflection Beam

Starting with v2.7, the `server-cuda` and `server-sagemaker-cuda` images serve Reflection's
[Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B) (`BeamForCausalLM`, 501B total / 23B active parameters). Upstream vLLM added Beam after
the commit these images are built from ([#62052](https://github.com/vllm-project/vllm/pull/62052),
[#62074](https://github.com/vllm-project/vllm/pull/62074)), so the images backport it. Beam is validated offline rather than in the regression suite
above.

Serve the NVFP4 checkpoint on four B200 or B300 GPUs with the `beam` reasoning and tool-call parsers:

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model reflection/Beam-1.0-501B-A23B-NVFP4 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser beam \
  --reasoning-parser beam \
  --host 0.0.0.0 --port 8000
```

- The first NVFP4 start can spend about 10 minutes autotuning the FlashInfer MoE kernels, so the command raises `VLLM_ENGINE_READY_TIMEOUT_S` from its
  600 s default.
- For DFlash2 speculative decoding, add
  `--speculative-config '{"method":"dflash","model":"reflection/Beam-1.0-501B-A23B-DFlash2","num_speculative_tokens":7}'`.
- The `beam` parser constrains every response to the model's output format. Set `-e VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=0` to turn this off.
- On {{ sagemaker }}, pass the same flags as `SM_VLLM_*` environment variables (see [Configuration](../configuration.md)).
- For the FP8 and BF16 checkpoints, Hopper GPUs, and tuning, see the [Beam recipe](https://recipes.vllm.ai/reflection/Beam-1.0-501B-A23B).

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
