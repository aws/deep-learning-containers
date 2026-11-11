# Amazon SageMaker AI Deployment

The image serves on **port 8080** and implements the {{ sagemaker }} contract itself: `GET /ping` reports readiness, and `POST /invocations` takes a
strands-decider request (see [API Endpoints](../index.md#api-endpoints)). There is no reverse proxy. The server is `strands-decider serve`, configured
through `DECIDER_*` environment variables on the container (see [Configuration](../configuration.md)).

A deployment has three steps: package the model with its base model, upload it to S3, and create the endpoint from that S3 location.

## Package the Model

The container loads the model offline from `/opt/ml/model`, so the model data holds the checkpoint and its base model in a `base/` folder, and the
checkpoint's config points at that folder (see [Packaging a Model](../models/index.md#packaging-a-model)). This script lays out a published model that
way. It needs `huggingface_hub` (`pip install huggingface_hub`):

```python
"""Download a strands-decider model and its base model, laid out for /opt/ml/model."""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = sys.argv[1]  # e.g. StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
OUT = Path(sys.argv[2])  # e.g. ./model
BASE_DIR = "/opt/ml/model/base"  # where the base model will be inside the container

checkpoint = OUT / MODEL.split("/")[-1]
snapshot_download(MODEL, local_dir=checkpoint)
config_path = next(
    p for p in (checkpoint / "strands_decider_config.json", checkpoint / "hobson_config.json") if p.exists()
)
config = json.loads(config_path.read_text())

# The base model at the revision the checkpoint was trained on: the config's, else provenance.json's.
base_model, base_revision = config["base_model"], config.get("base_revision")
provenance = checkpoint / "provenance.json"
if not base_revision and provenance.exists():
    meta = json.loads(provenance.read_text())
    if meta.get("base_model") == base_model:
        base_revision = meta.get("base_model_revision")
snapshot_download(base_model, revision=base_revision, local_dir=OUT / "base")

# Point the checkpoint at the local copy, so nothing asks the Hugging Face Hub at load time.
config["base_model"] = BASE_DIR
config["base_revision"] = None
config_path.write_text(json.dumps(config, indent=2))
```

Run it with an empty output folder: the container expects exactly one checkpoint folder next to `base/`, so packaging a second model into the same
folder makes it stop at start. Then pack the folder and upload it:

```bash
python prepare_model.py StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 ./model
tar -czf model.tar.gz -C model .
aws s3 cp model.tar.gz s3://<bucket>/strands-decider/strands-decider-2B-qwen3.5-v1-2610.tar.gz
```

The archive must have `base/` and the checkpoint folder at its root, not inside another folder. For the larger Gemma 4 models, you can skip the
archive: upload the folder as is (`aws s3 sync model s3://<bucket>/<prefix>/`) and pass it with `ModelDataSource` and `"CompressionType": "None"`
instead of `ModelDataUrl`, which saves the compression and extraction time.

## Real-Time Endpoint

```python
import boto3
import json

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")
REGION = boto3.session.Session().region_name

ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/strands-decider:serve-sagemaker-cuda-v1"
MODEL_DATA_URL = "s3://<bucket>/strands-decider/strands-decider-2B-qwen3.5-v1-2610.tar.gz"
NAME = "strands-decider-2b"

# 1. Model — the container finds the checkpoint under /opt/ml/model, so no environment is required
sm.create_model(
    ModelName=NAME,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_DATA_URL,
        # Optional, e.g. for the 26B-A4B model on a multi-GPU instance:
        # "Environment": {"DECIDER_DEVICE_MAP": "auto"},
    },
    ExecutionRoleArn=ROLE_ARN,
)

# 2. Endpoint config — pin the GPU host AMI, and allow time for the model load and warm-up
sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g6.xlarge",  # one L4; see Supported Models for the larger models
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
        "ContainerStartupHealthCheckTimeoutInSeconds": 900,
    }],
)

# 3. Endpoint
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)

# 4. Invoke
payload = {
    "state": "Help! My payouts have been failing for 3 days and nobody answers my emails!",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Route it.",
            "criteria": {"billing": "money, payouts, invoices", "technical": "bugs", "sales": "pricing"},
        },
    },
}
resp = smrt.invoke_endpoint(EndpointName=NAME, ContentType="application/json", Body=json.dumps(payload))
body = json.loads(resp["Body"].read())
print(body["answers"]["is_urgent"]["noul"])  # probability of yes, e.g. 0.86
print(body["answers"]["department"]["choice"], body["answers"]["department"]["probabilities"])

# 5. Cleanup
sm.delete_endpoint(EndpointName=NAME)
sm.delete_endpoint_config(EndpointConfigName=NAME)
sm.delete_model(ModelName=NAME)
```

The response has this shape (illustrative values; the numbers vary by model and GPU):

```json
{
  "model": "strands-decider-2B-qwen3.5-v1-2610",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.86},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.77, "technical": 0.21, "sales": 0.02},
      "confidence": 0.65
    }
  },
  "usage": {"input_tokens": 120, "output_tokens": 1},
  "latency_ms": 180.5
}
```

Callers need the IAM permission `sagemaker:InvokeEndpoint` on the endpoint. The server has no authentication of its own: {{ sagemaker }}'s request
signing is the access control.

## Test Locally

Before you create an endpoint, you can run the container on a GPU host the way {{ sagemaker }} runs it, with the packaged folder mounted at
`/opt/ml/model`. The server has no authentication, so publish the port on the loopback address only and do not expose it beyond the host:

```bash
docker run --rm --gpus all -p 127.0.0.1:8080:8080 -v "$PWD/model:/opt/ml/model:ro" \
  public.ecr.aws/deep-learning-containers/strands-decider:serve-sagemaker-cuda-v1 serve
# If Docker refuses to start the container with a `cuda>=` requirement error (some nvidia-container-toolkit versions),
# add -e NVIDIA_DISABLE_REQUIRE=true.

# In another shell, once the warm-up is done:
curl -s localhost:8080/ping
curl -s localhost:8080/invocations -H 'content-type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
```

## Notes

- **Pin the GPU host AMI.** The image runs PyTorch's CUDA 12.8 build on a CUDA 12.9 base. It is tested on `al2-ami-sagemaker-inference-gpu-3-1`
  (NVIDIA driver 550) and `al2023-ami-sagemaker-inference-gpu-4-1` (driver 580), set through `InferenceAmiVersion` on the production variant. On hosts
  whose driver is older than 575.57.08 (CUDA 12.9), the image turns on CUDA forward compatibility at start.
- **Allow for the warm-up.** Before `/ping` succeeds, the server loads the model and then answers every request shape once, so that no request waits
  for a kernel to compile. The model load and warm-up take about 70 seconds on the Qwen3.5 model. Set `ContainerStartupHealthCheckTimeoutInSeconds` to
  900, or more for the larger models. Without the warm-up (`DECIDER_WARMUP=0`), the first request on the Qwen3.5 model takes about 37 seconds.
- **One request runs at a time by default.** Real-time endpoints do not cap the number of requests in flight, and a burst of large requests run
  together can exhaust GPU memory. `DECIDER_MAX_CONCURRENCY=1` makes the others wait their turn instead (see
  [Configuration](../configuration.md#concurrency)). A queued request's wait counts toward the {{ sm_short }} invocation timeout of 60 seconds, so
  under sustained load add instances rather than raise the cap.
- **Ask many questions in one request.** The state is encoded once and each question adds only its own tokens, so several questions about one state
  cost little more than one. Warm requests in the tests take 0.1 to 0.5 seconds on every model and GPU.
- **Logs.** The container's output, including any error during the model load, is in CloudWatch Logs under `/aws/sagemaker/Endpoints/<endpoint name>`.

For all configuration options, see [Configuration](../configuration.md).
