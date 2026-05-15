# Amazon SageMaker AI Deployment

Deploy vLLM on Amazon SageMaker AI endpoints. The SageMaker image variant accepts model configuration via environment variables.

## SageMaker Python SDK v2

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="{{ images.latest_vllm_server_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={"SM_VLLM_MODEL": "openai/gpt-oss-20b"},
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
)

response = predictor.predict({
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256,
})
print(response)

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

## SageMaker Python SDK v3

```python
import json
import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

model = Model.create(
    model_name="vllm-model",
    primary_container=ContainerDefinition(
        image="{{ images.latest_vllm_server_sagemaker }}",
        environment={"SM_VLLM_MODEL": "openai/gpt-oss-20b"},
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
        "model": "openai/gpt-oss-20b",
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

## Boto3

```python
import json
import boto3

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")

sm.create_model(
    ModelName="vllm-model",
    PrimaryContainer={
        "Image": "{{ images.latest_vllm_server_sagemaker }}",
        "Environment": {"SM_VLLM_MODEL": "openai/gpt-oss-20b"},
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
        "model": "openai/gpt-oss-20b",
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

## SageMaker Features

The SageMaker vLLM image uses [standard-supervisor](https://github.com/aws/model-hosting-container-standards) to provide platform-specific
integrations:

**Process supervision and auto-recovery** — if the vLLM process crashes, it is automatically restarted (up to 3 retries by default). This prevents the
container from exiting on transient failures.

**Dynamic dependency installation** — bundle a `requirements.txt` with your model artifacts and dependencies are installed automatically before server
startup. No custom container needed.

**Custom handler support** — override the default `/ping` and `/invocations` endpoints with your own logic by placing a `model.py` in your model
artifacts:

```python
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import Request, Response

@sagemaker_standards.custom_invocation_handler
async def custom_invoke(request: Request) -> Response:
    body = await request.json()
    # your custom logic here
    return Response(content=json.dumps(result), media_type="application/json")
```

**LoRA adapter support** — dynamically load and route to LoRA adapters via request headers without restarting the server.

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 12.9 images. See
  [ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid values.
- Any `SM_VLLM_*` environment variable is converted to a `--<name>` vLLM server argument (e.g., `SM_VLLM_MAX_MODEL_LEN=4096` →
  `--max-model-len 4096`).

For all configuration options including server arguments, SageMaker environment variables, and standard-supervisor settings, see
[Configuration](../configuration.md).
