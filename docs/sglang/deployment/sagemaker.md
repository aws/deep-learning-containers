# Amazon SageMaker AI Deployment

Deploy SGLang on Amazon SageMaker AI endpoints. The SageMaker image variant accepts model configuration via environment variables and serves on port
8080\.

## Specifying the Model

The SageMaker image resolves the model in this order:

1. **`SM_SGLANG_MODEL_PATH` environment variable** — explicit Hugging Face ID or path
2. **`/opt/ml/model`** — when SageMaker mounts model artifacts via `ModelDataUrl` or `ModelDataSource`, the entrypoint uses this path by default

For gated models, also pass `HF_TOKEN`.

## SageMaker Python SDK v2

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="{{ images.latest_sglang_server_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={"SM_SGLANG_MODEL_PATH": "openai/gpt-oss-20b"},
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
    model_name="sglang-model",
    primary_container=ContainerDefinition(
        image="{{ images.latest_sglang_server_sagemaker }}",
        environment={"SM_SGLANG_MODEL_PATH": "openai/gpt-oss-20b"},
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

endpoint = Endpoint.create(endpoint_name="sglang-endpoint", endpoint_config_name="sglang-config")
endpoint.wait_for_status("InService")

smrt = boto3.client("sagemaker-runtime")
resp = smrt.invoke_endpoint(
    EndpointName="sglang-endpoint",
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
    ModelName="sglang-model",
    PrimaryContainer={
        "Image": "{{ images.latest_sglang_server_sagemaker }}",
        "Environment": {"SM_SGLANG_MODEL_PATH": "openai/gpt-oss-20b"},
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

sm.create_endpoint(EndpointName="sglang-endpoint", EndpointConfigName="sglang-config")
sm.get_waiter("endpoint_in_service").wait(EndpointName="sglang-endpoint")

resp = smrt.invoke_endpoint(
    EndpointName="sglang-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup
sm.delete_endpoint(EndpointName="sglang-endpoint")
sm.delete_endpoint_config(EndpointConfigName="sglang-config")
sm.delete_model(ModelName="sglang-model")
```

## Model Artifacts

When `ModelDataUrl` (or `ModelDataSource`) points to a tarball/S3 prefix, SageMaker mounts the contents at `/opt/ml/model`. The entrypoint defaults
`--model-path` to that location, so `SM_SGLANG_MODEL_PATH` can be omitted:

```text
model.tar.gz
├── config.json              # standard model files (Hugging Face layout)
├── tokenizer.json
└── *.safetensors
```

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 13 images. See
  [ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid values.
- Any `SM_SGLANG_*` environment variable is converted to a `--<name>` SGLang server argument (e.g., `SM_SGLANG_CONTEXT_LENGTH=4096` →
  `--context-length 4096`).

For all configuration options including server arguments and SageMaker environment variables, see [Configuration](../configuration.md).
