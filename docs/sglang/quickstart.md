# Quickstart

## Prerequisites

- An {{ aws }} account with appropriate permissions
- {{ aws }} CLI configured with your credentials
- A GPU instance ({{ ec2_short }} or {{ sm_short }})

## Pull the Image

### {{ ecr_public }} (Recommended)

No authentication required:

```bash
docker pull public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda
```

### Private {{ ecr }}

Requires authentication, uses a region-specific URI:

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-cuda
```

## Run on {{ ec2 }}

### Start the SGLang Server

```bash
docker run --gpus all -p 30000:30000 \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 30000
```

For gated models, add `-e HF_TOKEN=<your_hf_token>`:

```bash
docker run --gpus all -p 30000:30000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

### Send a Request

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
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
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-sagemaker-cuda",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={
        "SM_SGLANG_MODEL_PATH": "Qwen/Qwen3-0.6B",
        "HF_TOKEN": "<your_hf_token>",
    },
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
)

response = predictor.predict({
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256,
})
print(response)

# Cleanup when done
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

### SageMaker Python SDK v3

```python
import json

import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

model = Model.create(
    model_name="sglang-model",
    primary_container=ContainerDefinition(
        image="763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-sagemaker-cuda",
        environment={"SM_SGLANG_MODEL_PATH": "Qwen/Qwen3-0.6B"},
    ),
    execution_role_arn="arn:aws:iam::<account_id>:role/<role_name>",
)

ep_cfg = EndpointConfig.create(
    endpoint_config_name="sglang-config",
    production_variants=[
        ProductionVariant(
            variant_name="default",
            model_name="sglang-model",
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
        ),
    ],
)

endpoint = Endpoint.create(
    endpoint_name="sglang-endpoint",
    endpoint_config_name="sglang-config",
)
endpoint.wait_for_status("InService")

smrt = boto3.client("sagemaker-runtime")
resp = smrt.invoke_endpoint(
    EndpointName="sglang-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup when done
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
    ModelName="sglang-model",
    PrimaryContainer={
        "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-sagemaker-cuda",
        "Environment": {"SM_SGLANG_MODEL_PATH": "Qwen/Qwen3-0.6B"},
    },
    ExecutionRoleArn="arn:aws:iam::<account_id>:role/<role_name>",
)

sm.create_endpoint_config(
    EndpointConfigName="sglang-config",
    ProductionVariants=[{
        "VariantName": "default",
        "ModelName": "sglang-model",
        "InstanceType": "ml.g5.2xlarge",
        "InitialInstanceCount": 1,
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
    }],
)

sm.create_endpoint(
    EndpointName="sglang-endpoint",
    EndpointConfigName="sglang-config",
)

# Wait for endpoint to be InService
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName="sglang-endpoint")

resp = smrt.invoke_endpoint(
    EndpointName="sglang-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup when done
sm.delete_endpoint(EndpointName="sglang-endpoint")
sm.delete_endpoint_config(EndpointConfigName="sglang-config")
sm.delete_model(ModelName="sglang-model")
```

## Next Steps

- [Configuration](configuration.md) — Customize server arguments and environment variables
- [Deployment](deployment.md) — Production deployment patterns for {{ ec2 }}, {{ ecs }}, {{ eks }}, and {{ sagemaker }}
- [Supported Models](supported_models.md) — Full list of tested models and model families
