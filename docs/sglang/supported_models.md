# Supported Models

The SGLang AL2023 {{ dlc_short }} images support a wide range of models from the Hugging Face Hub. Any model compatible with the bundled SGLang
version can be served.

## Tested Models

### Image Validation ({{ ec2_short }})

| Model | Instance Type | TP | Extra Args |
| --- | --- | --- | --- |
| `Qwen3.5-0.8B` | `g6.xlarge` | 1 | `--context-length 4096 --dtype bfloat16` |

### Benchmark Models

| Model | Instance Type | TP | Extra Args |
| --- | --- | --- | --- |
| `GPT-OSS-20B` | `g6e.xlarge` | 1 | `--context-length 4096 --dtype bfloat16 --trust-remote-code` |
| `Qwen3.5-9B` | `g6e.xlarge` | 1 | `--context-length 4096 --dtype bfloat16` |
| `Qwen3-32B` | `p4d.24xlarge` | 4 | `--context-length 4096 --mem-fraction-static 0.85` |
| `Qwen3.5-27B-FP8` | `g6e.xlarge` | 1 | `--context-length 4096` |

### SageMaker Endpoint Test

| Model | Instance Type |
| --- | --- |
| `Qwen/Qwen3-0.6B` | `ml.g5.12xlarge` |

### Large Language Models

Any model compatible with the bundled SGLang version can be served. Common model families include:

| Model Family | Example Models | Min GPUs |
| --- | --- | --- |
| Llama 3.x | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | 1 |
| Qwen 2.5 / 3 | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-8B` | 1 |
| Mistral / Mixtral | `mistralai/Mistral-7B-Instruct-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` | 1 (7B), 2 (8x7B) |
| DeepSeek | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` | 1 (Lite), 8+ (V3) |
| Gemma | `google/gemma-3-1b-it`, `google/gemma-2-9b` | 1 |
| Phi | `microsoft/Phi-3-mini-4k-instruct`, `microsoft/phi-2` | 1 |

### Vision Language Models

SGLang supports vision language models through its OpenAI-compatible API. See
[upstream model support](https://docs.sglang.ai/references/supported_models.html) for the full list.

## Using Gated Models

Many models require accepting a license on Hugging Face. Set the `HF_TOKEN` environment variable:

=== "{{ ec2_short }}"

````
```bash
docker run --gpus all -p 30000:30000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000
```
````

=== "{{ sm_short }}"

````
```python
env = {
    "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-8B-Instruct",
    "HF_TOKEN": "<your_hf_token>",
}
```
````

## Using Custom Models

### From S3

```bash
docker run --gpus all -p 30000:30000 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path s3://<bucket>/<prefix>/my-model/
```

### From a Local Path

```bash
docker run --gpus all -p 30000:30000 \
  -v /local/models/my-model:/model \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path /model
```

## Upstream Model Support

For the complete list of supported architectures, see the
[SGLang supported models documentation](https://docs.sglang.ai/references/supported_models.html).

## See Also

- [Quickstart](quickstart.md) — run your first inference
- [Configuration](configuration.md) — server arguments and environment variables
- [Benchmarks](benchmarks.md) — throughput and latency by model
