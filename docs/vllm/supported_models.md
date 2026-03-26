# Supported Models

The vLLM {{ dlc_short }} images support a wide range of models from the Hugging Face Hub. Any model compatible with the bundled vLLM version can be
served.

## Tested Models

The following models are regularly tested with the vLLM {{ dlc_short }} images:

### Large Language Models

| Model Family | Example Models | Min GPUs |
| --- | --- | --- |
| Llama 3.1 | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct` | 1 (8B), 8 (70B) |
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` | 1 |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mixtral-8x7B-Instruct-v0.1` | 1 (7B), 2 (8x7B) |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-72B-Instruct` | 1 (7B), 8 (72B) |
| DeepSeek | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1` | 8+ |
| Gemma 2 | `google/gemma-2-9b-it`, `google/gemma-2-27b-it` | 1 (9B), 2 (27B) |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct` | 1 |

### Vision Language Models

| Model Family | Example Models | Min GPUs |
| --- | --- | --- |
| Llama 3.2 Vision | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 1 |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | 1 |
| Pixtral | `mistralai/Pixtral-12B-2409` | 1 |

## Using Gated Models

Many models require accepting a license on Hugging Face. Set the `HF_TOKEN` environment variable:

=== "{{ ec2_short }}"

````
```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  <image_uri> \
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
  <image_uri> \
  --model s3://<bucket>/<prefix>/my-model/
```

### From a Local Path

```bash
docker run --gpus all -p 8000:8000 \
  -v /local/models/my-model:/model \
  <image_uri> \
  --model /model
```

## Upstream Model Support

The {{ dlc_short }} images include the same model support as the bundled vLLM version. For the complete list of supported architectures, see the
[vLLM supported models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html).
