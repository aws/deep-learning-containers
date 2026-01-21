# Using {{ dlc }}

The following sections describe how to use {{ dlc }} to run sample code from each of the frameworks on {{ aws }} infrastructure.

## Use Cases

- For information on using {{ dlc }} with {{ sagemaker }}, see the
  [Use Your Own Algorithms or Models with {{ sagemaker }} Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html).

- To learn about using {{ dlc }} with {{ sagemaker }} HyperPod on {{ eks }}, see
  [Orchestrating SageMaker HyperPod clusters with {{ eks }} and {{ sagemaker }}](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html).

## Running on {{ sagemaker }}

### Using SageMaker Python SDK

#### Deploy an SGLang inference endpoint:

```python
from sagemaker.model import Model

model = Model(
    image_uri="{{ images.latest_sglang_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    env={
        "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-8B-Instruct",
        "HF_TOKEN": "<your_hf_token>",
    },
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
)
```

#### Deploy a vLLM inference endpoint:

```python
from sagemaker.model import Model

model = Model(
    image_uri="{{ images.latest_vllm_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    env={
        "SM_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        "HF_TOKEN": "<your_hf_token>",
    },
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
)
```

### Using Boto3

#### Deploy an SGLang inference endpoint:

```python
import boto3

sagemaker = boto3.client("sagemaker")

sagemaker.create_model(
    ModelName="sglang-model",
    PrimaryContainer={
        "Image": "{{ images.latest_sglang_sagemaker }}",
        "Environment": {
            "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-8B-Instruct",
            "HF_TOKEN": "<your_hf_token>",
        },
    },
    ExecutionRoleArn="arn:aws:iam::<account_id>:role/<role_name>",
)

sagemaker.create_endpoint_config(
    EndpointConfigName="sglang-endpoint-config",
    ProductionVariants=[
        {
            "VariantName": "default",
            "ModelName": "sglang-model",
            "InstanceType": "ml.g5.2xlarge",
            "InitialInstanceCount": 1,
            "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
        }
    ],
)

sagemaker.create_endpoint(
    EndpointName="sglang-endpoint",
    EndpointConfigName="sglang-endpoint-config",
)
```

#### Deploy a vLLM inference endpoint:

```python
import boto3

sagemaker = boto3.client("sagemaker")

sagemaker.create_model(
    ModelName="vllm-model",
    PrimaryContainer={
        "Image": "{{ images.latest_vllm_sagemaker }}",
        "Environment": {
            "SM_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
            "HF_TOKEN": "<your_hf_token>",
        },
    },
    ExecutionRoleArn="arn:aws:iam::<account_id>:role/<role_name>",
)

sagemaker.create_endpoint_config(
    EndpointConfigName="vllm-endpoint-config",
    ProductionVariants=[
        {
            "VariantName": "default",
            "ModelName": "vllm-model",
            "InstanceType": "ml.g5.2xlarge",
            "InitialInstanceCount": 1,
            "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
        }
    ],
)

sagemaker.create_endpoint(
    EndpointName="vllm-endpoint",
    EndpointConfigName="vllm-endpoint-config",
)
```

## Running on {{ ec2 }}

#### Running PyTorch Training Container on an {{ ec2_short }} Instance

```bash
# Run interactively
docker run -it --gpus all <account_id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag> bash

# Example: Run PyTorch container
docker run -it --gpus all {{ images.latest_pytorch_training_ec2 }} bash

# Mount local directories to persist data
docker run -it --gpus all -v /local/data:/data {{ images.latest_pytorch_training_ec2 }} bash
```

## Quick Links

- [Available Images](../reference/available_images.md) - Browse all container images
- [Support Policy](../reference/support_policy.md) - Framework versions and timelines
