# vLLM Inference

Pre-built Docker images for running [vLLM](https://docs.vllm.ai/) inference workloads on {{ aws }}. Built on Amazon Linux 2023 with CUDA 12.9 and
Python 3.12.

## Latest Announcements

**April 25, 2026** — vLLM 1.0.0 initial release on Amazon Linux 2023 with the simplified `server-cuda` tag family. Source is pinned to upstream vLLM
commit [6ef1efd5](https://github.com/vllm-project/vllm/commit/6ef1efd51f11106fc44deb9e7b2f5cd1247fc37e); wheel reports as `0.19.1+amzn2023.6ef1efd5`.

## Pull Commands

**EC2:**

```bash
docker pull {{ images.latest_vllm_server_ec2 }}
```

**SageMaker:**

```bash
docker pull {{ images.latest_vllm_server_sagemaker }}
```

See [Available Images](../reference/available_images.md) for all image URIs and [Getting Started](../get_started/index.md) for authentication
instructions.

## How We Build

The vLLM {{ dlc_short }} images are **curated builds**, not simple repackages of upstream releases:

- **Built from a chosen base reference** — a specific commit, release candidate, or point in vLLM's history — with targeted patches applied from
  upstream PRs, forks, and community contributions for new-model support, bug fixes, and performance improvements.
- **Opinionated testing** — validated against a selected suite of model-serving use cases relevant to {{ aws }} customers.
- **Faster access with higher confidence** — delivers the latest advancements while maintaining reliability for real-world workloads.

Each image ships with vLLM (OpenAI-compatible API server), PyTorch, CUDA, NCCL (multi-GPU), and security patches from
{{ aws }}.

For package versions included in each release, see the [Release Notes](../releasenotes/vllm-server/index.md).

## EC2 Deployment

The container runs the vLLM OpenAI-compatible API server on port 8000. Any `vllm serve` flag may be appended to `docker run`.

### Single GPU

```bash
docker run --gpus all -p 8000:8000 \
  {{ images.latest_vllm_server_ec2 }} \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 --port 8000
```

For gated models (Llama, etc.), pass `-e HF_TOKEN=<your_hf_token>`.

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256
  }'
```

### Multi-GPU (Tensor Parallelism)

For models that require multiple GPUs (e.g., 70B+):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  {{ images.latest_vllm_server_ec2 }} \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 --port 8000
```

Use `--ipc=host` for multi-GPU to enable shared memory between processes.

### Recommended Instance Types

| Instance Type | GPUs | GPU Memory | Use Case |
| --- | --- | --- | --- |
| `g5.xlarge` | 1× A10G (24 GB) | 24 GB | Small models (≤ 8B) |
| `g5.12xlarge` | 4× A10G (24 GB) | 96 GB | Medium models (8B–30B) |
| `p4d.24xlarge` | 8× A100 (40 GB) | 320 GB | Large models (30B–70B) |
| `p5.48xlarge` | 8× H100 (80 GB) | 640 GB | Very large models (70B+) |

### ECS / EKS

The container works as-is with ECS task definitions and Kubernetes pod specs. Key requirements:

- Request GPU resources (`resourceRequirements: [{type: GPU, value: "1"}]` for ECS, `resources.limits.nvidia.com/gpu: "1"` for EKS)
- Pass `--model`, `--host 0.0.0.0`, and `--port 8000` as container args
- Provide `HF_TOKEN` as an environment variable (or Kubernetes secret) for gated models
- Use `/health` on port 8000 for health checks

## SageMaker Deployment

### SageMaker Python SDK v2

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="{{ images.latest_vllm_server_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={"SM_VLLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
)

response = predictor.predict({
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256,
})
print(response)

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

GPU deploys require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 12.9 images. See
[ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid values.

### SageMaker Python SDK v3

```python
import json
import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

model = Model.create(
    model_name="vllm-model",
    primary_container=ContainerDefinition(
        image="{{ images.latest_vllm_server_sagemaker }}",
        environment={"SM_VLLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    ),
    execution_role_arn="arn:aws:iam::<account_id>:role/<role_name>",
)

ep_cfg = EndpointConfig.create(
    endpoint_config_name="vllm-config",
    production_variants=[
        ProductionVariant(
            variant_name="default",
            model_name="vllm-model",
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        ),
    ],
)

endpoint = Endpoint.create(endpoint_name="vllm-endpoint", endpoint_config_name="vllm-config")
endpoint.wait_for_status("InService")

smrt = boto3.client("sagemaker-runtime")
resp = smrt.invoke_endpoint(
    EndpointName="vllm-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup
endpoint.delete()
ep_cfg.delete()
model.delete()
```

### Boto3

```python
import json
import boto3

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")

sm.create_model(
    ModelName="vllm-model",
    PrimaryContainer={
        "Image": "{{ images.latest_vllm_server_sagemaker }}",
        "Environment": {"SM_VLLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    },
    ExecutionRoleArn="arn:aws:iam::<account_id>:role/<role_name>",
)

sm.create_endpoint_config(
    EndpointConfigName="vllm-config",
    ProductionVariants=[{
        "VariantName": "default",
        "ModelName": "vllm-model",
        "InstanceType": "ml.g5.2xlarge",
        "InitialInstanceCount": 1,
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
    }],
)

sm.create_endpoint(EndpointName="vllm-endpoint", EndpointConfigName="vllm-config")
sm.get_waiter("endpoint_in_service").wait(EndpointName="vllm-endpoint")

resp = smrt.invoke_endpoint(
    EndpointName="vllm-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup
sm.delete_endpoint(EndpointName="vllm-endpoint")
sm.delete_endpoint_config(EndpointConfigName="vllm-config")
sm.delete_model(ModelName="vllm-model")
```

## Configuration

### SageMaker Environment Variables

Any `SM_VLLM_*` env var is converted to a `--<name>` vLLM server argument (e.g., `SM_VLLM_MAX_MODEL_LEN=4096` → `--max-model-len 4096`).

| Variable | Description | Required |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID from Hugging Face Hub or S3 path | Yes |
| `HF_TOKEN` | Hugging Face token for gated models | For gated models |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | No |
| `SM_VLLM_MAX_MODEL_LEN` | Maximum sequence length | No |
| `SM_VLLM_ENFORCE_EAGER` | Set to `true` to disable CUDA graphs | No |

### Common Server Arguments (EC2)

| Argument | Description | Default |
| --- | --- | --- |
| `--model` | Model ID or path | Required |
| `--host` | Bind address | `localhost` |
| `--port` | Server port | `8000` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `1` |
| `--max-model-len` | Maximum sequence length | Model default |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.9` |
| `--enforce-eager` | Disable CUDA graph for debugging | `false` |
| `--quantization` | Quantization method (awq, gptq, fp8, …) | None |
| `--dtype` | Model data type (auto, float16, bfloat16) | `auto` |

For the complete list, see the [vLLM engine arguments documentation](https://docs.vllm.ai/en/latest/configuration/engine_args.html) and
[vLLM environment variables](https://docs.vllm.ai/en/latest/configuration/env_vars.html).

## Supported Models

The following models have been validated on the bundled vLLM version:

| Model Family | Example Models | Tested Instance |
| --- | --- | --- |
| Qwen3 | `Qwen/Qwen3-32B` | `p4d.24xlarge` |
| Llama 3.x | `meta-llama/Llama-3.3-70B-Instruct` | `p4d.24xlarge` |
| Gemma 4 | `google/gemma-4-e4b-it`, `google/gemma-4-31b-it` | `g6e.xlarge`, `g6e.12xlarge` |
| Qwen3-ASR (speech recognition) | `Qwen/Qwen3-ASR-1.7B` | `g6e.12xlarge` |
| GPT-OSS | `openai/gpt-oss-20b` | `g6e.xlarge` |
| Qwen3.5 (MoE, FP8) | `Qwen/Qwen3.5-35B-A3B-FP8` | `g6e.12xlarge` |
| Gemma 4 (MoE) | `google/gemma-4-26b-a4b-it` | `g6e.12xlarge` |

### Using Custom Models

**From S3:**

```bash
docker run --gpus all -p 8000:8000 \
  -e AWS_DEFAULT_REGION=us-west-2 \
  {{ images.latest_vllm_server_ec2 }} \
  --model s3://<bucket>/<prefix>/my-model/
```

**From a local path:**

```bash
docker run --gpus all -p 8000:8000 \
  -v /local/models/my-model:/model \
  {{ images.latest_vllm_server_ec2 }} \
  --model /model
```

## Release Notes

See [vLLM Release Notes](../releasenotes/vllm-server/index.md) for version history and changelogs.

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
