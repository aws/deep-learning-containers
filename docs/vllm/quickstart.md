# Quickstart

Get a vLLM {{ dlc_short }} container running and serve your first model.

## Prerequisites

- An {{ aws }} account with appropriate permissions
- {{ aws }} CLI configured with your credentials
- A GPU instance ({{ ec2_short }} or {{ sm_short }})

## Pull the Image

### {{ ecr_public }} (Recommended)

No authentication required:

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda
```

### Private {{ ecr }}

Requires authentication, uses a region-specific URI:

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:server-cuda
```

For version pinning options (e.g., `server-cuda-v1.0.0`), see [Versioning](versioning.md).

## Run on {{ ec2 }}

### Start the vLLM Server

```bash
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000
```

For gated models (e.g., Llama), add `-e HF_TOKEN=<your_hf_token>`:

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Send a Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256
  }'
```

## Run on {{ sagemaker }}

### SageMaker Python SDK v2

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="{{ images.latest_vllm_sagemaker }}",
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
```

### SageMaker Python SDK v3

```python
import json

import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

model = Model.create(
    model_name="vllm-model",
    primary_container=ContainerDefinition(
        image="{{ images.latest_vllm_sagemaker }}",
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

endpoint = Endpoint.create(
    endpoint_name="vllm-endpoint",
    endpoint_config_name="vllm-config",
)
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
        "Image": "{{ images.latest_vllm_sagemaker }}",
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

sm.create_endpoint(
    EndpointName="vllm-endpoint",
    EndpointConfigName="vllm-config",
)

# Wait for endpoint to be InService, then invoke:
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
```

## Next Steps

- [Configuration](configuration.md) — Tune engine arguments and environment variables
- [Deployment](deployment.md) — Multi-GPU, multi-node, and production setups
- [Benchmarks](benchmarks.md) — Performance numbers on {{ aws }} GPU instances
