# Deployment

Deploy vLLM {{ dlc_short }} images across {{ aws }} compute platforms.

## {{ ec2 }}

### Single GPU

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000
```

### Multi-GPU (Tensor Parallelism)

For models that require multiple GPUs (e.g., 70B+ parameter models):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 --port 8000
```

> **Tip:** Use `--ipc=host` for multi-GPU setups to enable shared memory between processes.

### Recommended Instance Types

| Instance Type | GPUs | GPU Memory | Use Case |
| --- | --- | --- | --- |
| `g5.xlarge` | 1x A10G (24 GB) | 24 GB | Small models (≤ 8B) |
| `g5.12xlarge` | 4x A10G (24 GB) | 96 GB | Medium models (8B–30B) |
| `p4d.24xlarge` | 8x A100 (40 GB) | 320 GB | Large models (30B–70B) |
| `p5.48xlarge` | 8x H100 (80 GB) | 640 GB | Very large models (70B+) |

## {{ sagemaker }}

### Using SageMaker Python SDK

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:server-sagemaker-cuda",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={
        "SM_VLLM_MODEL": "meta-llama/Llama-3.1-70B-Instruct",
        "HF_TOKEN": "<your_hf_token>",
        "SM_VLLM_TENSOR_PARALLEL_SIZE": "8",
        "SM_VLLM_MAX_MODEL_LEN": "8192",
    },
)

predictor = model.deploy(
    instance_type="ml.p4d.24xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
)

response = predictor.predict({
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 256,
})
print(response)

# Cleanup when done
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

### Using Boto3

```python
import json

import boto3

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")

sm.create_model(
    ModelName="vllm-model",
    PrimaryContainer={
        "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:server-sagemaker-cuda",
        "Environment": {
            "SM_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
            "HF_TOKEN": "<your_hf_token>",
        },
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

# Wait for endpoint to be InService
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName="vllm-endpoint")

resp = smrt.invoke_endpoint(
    EndpointName="vllm-endpoint",
    ContentType="application/json",
    Body=json.dumps({
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
)
print(json.loads(resp["Body"].read()))

# Cleanup when done
sm.delete_endpoint(EndpointName="vllm-endpoint")
sm.delete_endpoint_config(EndpointConfigName="vllm-config")
sm.delete_model(ModelName="vllm-model")
```

## {{ ecs }} / {{ eks }}

### {{ ecs_short }} Task Definition

```json
{
  "family": "vllm-inference",
  "networkMode": "host",
  "containerDefinitions": [
    {
      "name": "vllm",
      "image": "public.ecr.aws/deep-learning-containers/vllm:server-cuda",
      "command": [
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "environment": [
        {"name": "HF_TOKEN", "value": "<your_hf_token>"}
      ],
      "memoryReservation": 14000,
      "portMappings": [{"containerPort": 8000, "hostPort": 8000}],
      "resourceRequirements": [
        {"type": "GPU", "value": "1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/vllm-inference",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "vllm",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
```

### {{ eks_short }} Pod Spec

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-inference
spec:
  containers:
    - name: vllm
      image: public.ecr.aws/deep-learning-containers/vllm:server-cuda
      args:
        - "--model"
        - "meta-llama/Llama-3.1-8B-Instruct"
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "8000"
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
      ports:
        - containerPort: 8000
      resources:
        limits:
          nvidia.com/gpu: "1"
```

## Multi-Node Deployment

For models that exceed the memory of a single instance, use pipeline parallelism across nodes with EFA networking:

```bash
docker run --gpus all --ipc=host --network=host \
  --privileged \
  -e HF_TOKEN=<your_hf_token> \
  -e NCCL_DEBUG=INFO \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --host 0.0.0.0 --port 8000
```

> **Note:** Multi-node deployments require EFA-enabled instances (e.g., `p4d.24xlarge`, `p5.48xlarge`) and appropriate security group configuration
> for EFA traffic.

## Health Checks

The vLLM server exposes a health endpoint:

```bash
curl http://localhost:8000/health
```

Use this for load balancer health checks, {{ ecs_short }} health checks, and Kubernetes liveness/readiness probes.
