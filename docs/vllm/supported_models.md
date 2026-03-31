# Supported Models

The vLLM {{ dlc_short }} images support a wide range of models from the Hugging Face Hub. Any model compatible with the bundled vLLM version can be
served.

## Tested Models

The following models are tested in {{ dlc_short }} CI. Models from the
[upstream vLLM test registry](https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py) are run as image validation tests, while
benchmark models are tested for throughput and latency thresholds.

### Image Validation (EC2)

These models run as part of the {{ dlc_short }} image build validation:

| Test | Model | Task |
| --- | --- | --- |
| Text generation | `facebook/opt-125m` | Offline inference |
| Chat | `Qwen/Qwen2.5-0.5B-Instruct` | Chat completion |
| Audio | `fixie-ai/ultravox-v0_5-llama-3_2-1b` | Audio language |
| Vision | `llava-hf/llava-1.5-7b-hf` | Vision language |
| Vision (multi-image) | `microsoft/Phi-3.5-vision-instruct` | Multi-image inference |
| Whisper | `openai/whisper-large-v3-turbo` | Speech-to-text |
| Classification | `jason9693/Qwen2.5-1.5B-apeach` | Sequence classification |
| Embedding | `intfloat/e5-mistral-7b-instruct` | Text embedding |
| Scoring | `BAAI/bge-reranker-v2-m3` | Cross-encoder scoring |
| Speculative (Eagle) | `meta-llama/Meta-Llama-3-8B-Instruct` | Speculative decoding |
| Speculative (Eagle3) | `meta-llama/Llama-3.1-8B-Instruct` | Speculative decoding |
| Prefix caching | `facebook/opt-125m` | Prefix caching |

### Benchmark Models

These models are benchmarked for throughput and latency in CI (see [Benchmarks](benchmarks.md)):

| Model | Instance Types | TP | Quantization |
| --- | --- | --- | --- |
| `gpt-oss-20b` | `g6e.xlarge` | 1 | MXFP4 |
| `qwen3.5-9b` | `g6.xlarge` | 1 | None |
| `llama-3.3-70b` | `p4d.24xlarge`, `g6e.12xlarge` | 4 | None |
| `qwen3-32b` | `p4d.24xlarge` | 4 | None |
| `qwen3-coder-next-fp8` | `p4d.24xlarge`, `g6e.12xlarge` | 4 | FP8 |
| `qwen3.5-27b-fp8` | `p4d.24xlarge`, `g6e.12xlarge` | 4 | FP8 |
| `qwen3.5-35b-a3b-fp8` | `p4d.24xlarge`, `g6e.12xlarge` | 4 | FP8 |

### Large Language Models

Any model compatible with the bundled vLLM version can be served. Common model families include:

| Model Family | Example Models | Min GPUs |
| --- | --- | --- |
| Llama 3.x | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | 1 |
| Qwen 2.5 / 3 | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-8B` | 1 |
| Mistral / Mixtral | `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` | 1 (7B), 2 (8x7B) |
| DeepSeek | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` | 1 (Lite), 8+ (V3) |
| Gemma | `google/gemma-3-1b-it`, `google/gemma-2-9b` | 1 |
| Phi | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/phi-2` | 1 |

### Vision Language Models

| Model Family | Example Models | Min GPUs |
| --- | --- | --- |
| LLaVA | `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | 1 |
| Qwen2-VL / Qwen2.5-VL | `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct` | 1 |
| Pixtral | `mistralai/Pixtral-12B-2409` | 1 |
| Phi-3 Vision | `microsoft/Phi-3.5-vision-instruct` | 1 |
| InternVL | `OpenGVLab/InternVL2-1B`, `OpenGVLab/InternVL3-1B` | 1 |

## Using Gated Models

Many models require accepting a license on Hugging Face. Set the `HF_TOKEN` environment variable:

=== "{{ ec2_short }}"

````
```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.1-8B-Instruct
```
````

=== "{{ sm_short }}"

````
```python
env = {
    "SM_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    "HF_TOKEN": "<your_hf_token>",
}
```
````

## Using Custom Models

### From S3

```bash
docker run --gpus all -p 8000:8000 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model s3://<bucket>/<prefix>/my-model/
```

### From a Local Path

```bash
docker run --gpus all -p 8000:8000 \
  -v /local/models/my-model:/model \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model /model
```

## Upstream Model Support

The {{ dlc_short }} images include the same model support as the bundled vLLM version. For the complete list of supported architectures, see the
[vLLM supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).
