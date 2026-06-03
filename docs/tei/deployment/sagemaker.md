# Amazon SageMaker AI Deployment

Deploy TEI on Amazon SageMaker AI endpoints. The container accepts model configuration via environment variables and serves on port 8080.

## Specifying the Model

Set **`HF_MODEL_ID`** to a Hugging Face Hub model ID (e.g. `BAAI/bge-base-en-v1.5`). The container exits at startup if it is not set. For gated or
private models, also pass `HF_TOKEN`.

To serve model artifacts from S3 instead, provide them via `ModelDataUrl`/`ModelDataSource` — SageMaker mounts them at `/opt/ml/model` — and set
`HF_MODEL_ID=/opt/ml/model`.

## SageMaker Python SDK v2

```python
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

model = HuggingFaceModel(
    # Resolves the TEI GPU image for your session's region.
    # For CPU instances, use the "huggingface-tei-cpu" backend instead.
    image_uri=get_huggingface_llm_image_uri("huggingface-tei", version="1.8.2"),
    role="arn:aws:iam::<account_id>:role/<role_name>",
    env={"HF_MODEL_ID": "BAAI/bge-base-en-v1.5"},
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
)

response = predictor.predict({"inputs": "What is deep learning?"})
print(response)  # [[0.0123, -0.0456, ...]]

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

## SageMaker Python SDK v3

```python
import json

import boto3
from sagemaker.core import image_uris
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

# Requires a configured AWS region (always set inside SageMaker environments).
region = boto3.session.Session().region_name

# Resolves the TEI GPU image for `region`. For CPU instances, use the
# "huggingface-tei-cpu" framework instead.
image = image_uris.retrieve(
    framework="huggingface-tei",
    region=region,
    version="1.8.2",
    image_scope="inference",
    instance_type="ml.g5.xlarge",
)

model = Model.create(
    model_name="tei-model",
    primary_container=ContainerDefinition(
        image=image,
        environment={"HF_MODEL_ID": "BAAI/bge-base-en-v1.5"},
    ),
    execution_role_arn="arn:aws:iam::<account_id>:role/<role_name>",
    region=region,
)

ep_cfg = EndpointConfig.create(
    endpoint_config_name="tei-config",
    production_variants=[
        ProductionVariant(
            variant_name="default",
            model_name="tei-model",
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
        ),
    ],
    region=region,
)

endpoint = Endpoint.create(
    endpoint_name="tei-endpoint",
    endpoint_config_name="tei-config",
    region=region,
)
endpoint.wait_for_status("InService")

smrt = boto3.client("sagemaker-runtime", region_name=region)
resp = smrt.invoke_endpoint(
    EndpointName="tei-endpoint",
    ContentType="application/json",
    Body=json.dumps({"inputs": "What is deep learning?"}),
)
print(json.loads(resp["Body"].read()))  # [[0.0123, -0.0456, ...]]

# Cleanup
endpoint.delete()
ep_cfg.delete()
model.delete()
```

## Boto3

```python
import json

import boto3

# TEI 1.8.2 GPU image in us-east-1. The image URI is region-specific (both the
# account ID and the region segment) — for other regions, replace `region` and
# `image` together using the region table: ../index.md#images
region = "us-east-1"
image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/tei:2.0.1-tei1.8.2-gpu-py310-cu122-ubuntu22.04"

# Clients are pinned to the same region as the image so the model, endpoint,
# and container stay co-located.
sm = boto3.client("sagemaker", region_name=region)
smrt = boto3.client("sagemaker-runtime", region_name=region)

sm.create_model(
    ModelName="tei-model-boto3",
    PrimaryContainer={
        "Image": image,
        "Environment": {"HF_MODEL_ID": "BAAI/bge-base-en-v1.5"},
    },
    ExecutionRoleArn="arn:aws:iam::<account_id>:role/<role_name>",
)

sm.create_endpoint_config(
    EndpointConfigName="tei-config-boto3",
    ProductionVariants=[{
        "VariantName": "default",
        "ModelName": "tei-model-boto3",
        "InstanceType": "ml.g5.xlarge",
        "InitialInstanceCount": 1,
    }],
)

sm.create_endpoint(
    EndpointName="tei-endpoint-boto3",
    EndpointConfigName="tei-config-boto3",
)
sm.get_waiter("endpoint_in_service").wait(EndpointName="tei-endpoint-boto3")

resp = smrt.invoke_endpoint(
    EndpointName="tei-endpoint-boto3",
    ContentType="application/json",
    Body=json.dumps({"inputs": "What is deep learning?"}),
)
print(json.loads(resp["Body"].read()))  # [[0.0123, -0.0456, ...]]

# Cleanup
sm.delete_endpoint(EndpointName="tei-endpoint-boto3")
sm.delete_endpoint_config(EndpointConfigName="tei-config-boto3")
sm.delete_model(ModelName="tei-model-boto3")
```

## Configuration

TEI is configured through environment variables that mirror its [CLI arguments](https://huggingface.co/docs/text-embeddings-inference/cli_arguments).
Commonly used options:

| Environment Variable | Purpose |
| --- | --- |
| `HF_MODEL_ID` | Hugging Face Hub model ID to serve |
| `HF_TOKEN` | Hub token for gated/private models |
| `MAX_CONCURRENT_REQUESTS` | Maximum number of concurrent requests |
| `MAX_BATCH_TOKENS` | Token budget per batch (controls throughput vs. memory) |
| `MAX_CLIENT_BATCH_SIZE` | Maximum number of inputs in a single request |
| `MAX_BATCH_REQUESTS` | Optional cap on requests per batch |
| `DTYPE` | Activation dtype (`float16`, `float32`) |
| `POOLING` | Override pooling method (`cls`, `mean`, `splade`, `last-token`) |
| `AUTO_TRUNCATE` | Truncate inputs longer than the model's maximum sequence length instead of rejecting them |

## Notes

- Choose the GPU image (`tei`) for GPU instances (e.g. `ml.g5`, `ml.g6`) and the CPU image (`tei-cpu`) for CPU instances (e.g. `ml.c6i`, `ml.c7i`,
  `ml.m6i`). For the CPU SDK v2 helper, use `get_huggingface_llm_image_uri("huggingface-tei-cpu", version="1.8.2")`. Embedding models are small
  relative to LLMs — single-GPU and CPU instances are usually sufficient.
- Rerankers and classifiers deploy with the same images and code — only `HF_MODEL_ID` and the request payload change (e.g. `BAAI/bge-reranker-v2-m3`
  with `{"query": "...", "texts": ["...", "..."]}`). See [API Endpoints](../index.md#api-endpoints) for the payload per model type.
- Inputs longer than the model's maximum sequence length are rejected unless you request truncation, either per request
  (`{"inputs": "...", "truncate": true}`) or globally (`AUTO_TRUNCATE=true`).
