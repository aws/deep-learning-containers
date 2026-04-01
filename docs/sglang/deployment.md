# Deployment

## {{ ec2 }}

### Single GPU

```bash
docker run --gpus all -p 30000:30000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000
```

### Multi-GPU (Tensor Parallelism)

For models that require multiple GPUs (e.g., 30B+ parameter models):

```bash
docker run --gpus all --ipc=host -p 30000:30000 \
  -e HF_TOKEN=<your_hf_token> \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp 8 \
  --host 0.0.0.0 --port 30000
```

> **Tip:** Use `--ipc=host` for multi-GPU setups to enable shared memory between processes.

### Recommended Instance Types

| Instance Type | GPUs | GPU Memory | Use Case |
| --- | --- | --- | --- |
| `g6.xlarge` | 1x L4 (24 GB) | 24 GB | Small models (≤ 9B) |
| `g6e.xlarge` | 1x L40S (48 GB) | 48 GB | Medium models (9B–27B) |
| `g6e.12xlarge` | 4x L40S (48 GB) | 192 GB | Large models (27B–70B) |
| `p4d.24xlarge` | 8x A100 (40 GB) | 320 GB | Large models (32B–70B) |
| `p5.48xlarge` | 8x H100 (80 GB) | 640 GB | Very large models (70B+) |

## {{ sagemaker }}

### Using SageMaker Python SDK

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-sagemaker-cuda",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={
        "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-70B-Instruct",
        "HF_TOKEN": "<your_hf_token>",
        "SM_SGLANG_TENSOR_PARALLEL_SIZE": "8",
        "SM_SGLANG_CONTEXT_LENGTH": "8192",
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

For {{ sagemaker }} environment variable configuration, see [Configuration](configuration.md).

### Using Boto3

```python
import json

import boto3

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")

sm.create_model(
    ModelName="sglang-model",
    PrimaryContainer={
        "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:server-amzn2023-sagemaker-cuda",
        "Environment": {
            "SM_SGLANG_MODEL_PATH": "meta-llama/Llama-3.1-8B-Instruct",
            "HF_TOKEN": "<your_hf_token>",
        },
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
        "model": "meta-llama/Llama-3.1-8B-Instruct",
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

## {{ ecs }} / {{ eks }}

### {{ ecs_short }} Task Definition

```json
{
  "family": "sglang-inference",
  "networkMode": "host",
  "containerDefinitions": [
    {
      "name": "sglang",
      "image": "public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda",
      "command": [
        "--model-path", "meta-llama/Llama-3.1-8B-Instruct",
        "--host", "0.0.0.0",
        "--port", "30000"
      ],
      "environment": [
        {"name": "HF_TOKEN", "value": "<your_hf_token>"}
      ],
      "memoryReservation": 14000,
      "resourceRequirements": [
        {"type": "GPU", "value": "1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sglang-inference",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "sglang",
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
  name: sglang-inference
spec:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
  containers:
    - name: sglang
      image: public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda
      args:
        - "--model-path"
        - "meta-llama/Llama-3.1-8B-Instruct"
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "30000"
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
      ports:
        - containerPort: 30000
      resources:
        limits:
          nvidia.com/gpu: "1"
```

## Multi-Node Deployment

For models that exceed the memory of a single instance, use tensor parallelism across nodes with EFA networking. SGLang uses `--nnodes`,
`--node-rank`, and `--dist-init-addr` to coordinate multi-node serving:

```bash
# Node 0 (head)
docker run --gpus all --ipc=host --network=host \
  --privileged \
  -e HF_TOKEN=<your_hf_token> \
  -e NCCL_DEBUG=INFO \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp 8 --nnodes 2 --node-rank 0 \
  --dist-init-addr <head_node_ip>:5000 \
  --host 0.0.0.0 --port 30000

# Node 1 (worker)
docker run --gpus all --ipc=host --network=host \
  --privileged \
  -e HF_TOKEN=<your_hf_token> \
  -e NCCL_DEBUG=INFO \
  public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda \
  --model-path meta-llama/Llama-3.1-405B-Instruct \
  --tp 8 --nnodes 2 --node-rank 1 \
  --dist-init-addr <head_node_ip>:5000 \
  --host 0.0.0.0 --port 30000
```

> **Note:** Multi-node deployments require EFA-enabled instances (e.g., `p4d.24xlarge`, `p5.48xlarge`) and appropriate security group configuration
> for EFA traffic.

## Health Checks

The SGLang server exposes a health endpoint:

```bash
curl http://localhost:30000/health
```

Use this for load balancer health checks, {{ ecs_short }} health checks, and Kubernetes liveness/readiness probes.

## See Also

- [Configuration](configuration.md) — server arguments and environment variables
- [Benchmarks](benchmarks.md) — throughput and latency by model
- [Supported Models](supported_models.md) — tested models and compatibility
