# Supported Models

The image serves the published strands-decider models, which live in the [StrandsAgents](https://huggingface.co/StrandsAgents) organization on Hugging
Face. Each model is a checkpoint (a LoRA adapter, a small pointer head, the tokenizer, and a config) on top of a base model that is stored separately.
Each container serves one model.

| Model | Base model family | GPU memory for the weights (bf16) | Tested on |
| --- | --- | --- | --- |
| [`strands-decider-2B-qwen3.5-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) | Qwen3.5 2B | about 4.5 GB | One L4, and a {{ sagemaker }} endpoint |
| [`strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610) | Gemma 4 E2B | about 9 GB | One L4 |
| [`strands-decider-E4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610) | Gemma 4 E4B | about 15 GB | One L4 |
| [`strands-decider-12B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610) | Gemma 4 12B | about 23 GB | One L40S |
| [`strands-decider-26B-A4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610) | Gemma 4 26B-A4B | about 51 GB | Split across L40S GPUs with `DECIDER_DEVICE_MAP=auto` |

The memory figures are for the weights only, from the strands-decider documentation. Activations come on top and grow with the state length and the
number of questions per forward pass. For accuracy and calibration of each model, see the strands-decider
[README](https://github.com/strands-labs/strands-decider#gemma-4-models). The earlier 2B models (`strands-decider-2B-hobson-v21` and `-v19`) have the
same architecture as the Qwen3.5 2B model, and the image reads their `hobson_config.json`, but they are not part of the image's test suite.

Image input (`serve --vision`) is not supported: the image does not include the `vision` extra of strands-decider.

## Choosing an Instance

The GPU tests run each model on the GPUs below. Every test checks the answers for each question type, a request of 40 questions, a burst of 64 large
requests, and that `/ping` answers within 2 seconds under load.

| Model | Tested GPU | {{ sagemaker }} instance |
| --- | --- | --- |
| 2B (Qwen3.5), E2B, E4B | One NVIDIA L4 (24 GB) | `ml.g6.xlarge` |
| 12B | One NVIDIA L40S (48 GB) | `ml.g6e.xlarge` |
| 26B-A4B | Four or eight NVIDIA L40S, with `DECIDER_DEVICE_MAP=auto` | `ml.g6e.12xlarge` (four L40S) or `ml.g6e.48xlarge` (eight L40S) |

The {{ sagemaker }} endpoint test deploys the 2B model on `ml.g6.xlarge`, with `ml.g6.2xlarge`, `ml.g6.4xlarge`, `ml.g6e.xlarge`, and `ml.g5.2xlarge`
(one A10G) as fallbacks.

`DECIDER_DEVICE_MAP=auto` splits a model larger than one GPU across all the GPUs of the instance. The layers run one GPU after another, so the split
adds memory, not speed. A single GPU with more than 51 GB of memory, such as the H100 in `ml.p5.4xlarge`, should hold the 26B-A4B model without a
split; that setup is not tested. The image compiles its CUDA kernels for A100, A10G, L4, L40S, and H100 GPUs (compute capabilities 8.0, 8.6, 8.9, and
9.0).

## Packaging a Model

The image runs offline (`HF_HUB_OFFLINE=1`), so the model you give to {{ sagemaker }} must hold both the checkpoint and its base model.
{{ sagemaker }} copies it to `/opt/ml/model`, which must look like this:

```text
/opt/ml/model/
├── strands-decider-2B-qwen3.5-v1-2610/   # the checkpoint
│   ├── strands_decider_config.json        # "base_model": "/opt/ml/model/base"
│   ├── lora/
│   ├── head.safetensors
│   └── tokenizer.json, tokenizer_config.json, ...
└── base/                                  # the base model, e.g. Qwen/Qwen3.5-2B-Base
```

A published checkpoint names its base model by its Hugging Face id, which cannot load offline. Set `base_model` in the checkpoint's
`strands_decider_config.json` (or `hobson_config.json`) to `/opt/ml/model/base`. When a `base/` folder sits next to the checkpoint and the config does
not name it, the container stops at start with a message that says so.

The container finds the checkpoint by itself: `/opt/ml/model` when it holds the checkpoint files, otherwise its one subfolder that does. With more
than one checkpoint, set `DECIDER_CHECKPOINT` (see [Configuration](../configuration.md)). The checkpoint folder name is the `model` value in responses
and in `/ping`.

[{{ sagemaker }} Deployment](../deployment/sagemaker.md#package-the-model) has a script that downloads a model and its base model at the right
revision and writes this layout.

## Full Reference

- [strands-decider](https://github.com/strands-labs/strands-decider)
- [strands-decider inference documentation](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md) (model artifact, request
  format, `serve` options)
- [StrandsAgents on Hugging Face](https://huggingface.co/StrandsAgents)
