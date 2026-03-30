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

### Using SageMaker Python SDK

```python
from sagemaker.model import Model

model = Model(
    image_uri="{{ images.latest_vllm_sagemaker }}",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    env={
        "SM_VLLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
)
```

### Send a Request

```python
import json

response = predictor.predict(
    json.dumps({
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [{"role": "user", "content": "What is deep learning?"}],
        "max_tokens": 256,
    }),
    initial_args={"ContentType": "application/json"},
)
print(json.loads(response))
```

## Next Steps

- [Configuration](configuration.md) — Tune engine arguments and environment variables
- [Deployment](deployment.md) — Multi-GPU, multi-node, and production setups
- [Benchmarks](benchmarks.md) — Performance numbers on {{ aws }} GPU instances
