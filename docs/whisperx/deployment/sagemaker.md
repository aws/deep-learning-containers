# Amazon SageMaker AI Deployment

The {{ sagemaker }} image (`3.8.6-cu128-amzn2023-sagemaker`) serves on **port 8080** and exposes `POST /invocations`. The endpoint expects
`multipart/form-data`: the client builds the multipart body — the audio as the `file` part plus optional string form fields — and SageMaker passes the
`ContentType` header (including the boundary) through to the container unchanged.

Both examples below use `boto3` because the payload is multipart. Every GPU variant must set `InferenceAmiVersion` — see [Notes](#notes).

## Specifying the Model

The SageMaker image resolves the model in this order:

1. **`WHISPERX_DEFAULT_MODEL` environment variable** — an explicit Whisper model id or path
2. **`/opt/ml/model`** — a model staged via `ModelDataUrl` is auto-detected and served offline
3. **image default** (`large-v2`)

## Real-Time Endpoint

```python
import boto3
import uuid

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")

ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = "public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023-sagemaker"
NAME = "whisperx-realtime"


def build_multipart(audio_path, fields):
    """Build a multipart/form-data body with the audio as the `file` part."""
    boundary = uuid.uuid4().hex
    crlf = b"\r\n"
    body = b""
    for key, value in fields.items():
        body += b"--" + boundary.encode() + crlf
        body += f'Content-Disposition: form-data; name="{key}"'.encode() + crlf + crlf
        body += str(value).encode() + crlf
    with open(audio_path, "rb") as f:
        audio = f.read()
    body += b"--" + boundary.encode() + crlf
    body += b'Content-Disposition: form-data; name="file"; filename="audio.wav"' + crlf
    body += b"Content-Type: audio/wav" + crlf + crlf + audio + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return body, f"multipart/form-data; boundary={boundary}"


# 1. Model
sm.create_model(
    ModelName=NAME,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "Environment": {"WHISPERX_DEFAULT_MODEL": "large-v2"},  # optional
    },
    ExecutionRoleArn=ROLE_ARN,
)

# 2. Endpoint config — the GPU AMI pin is REQUIRED (see Notes)
sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g4dn.xlarge",  # or ml.g5.2xlarge
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
        # Models warm before /ping, so allow a generous startup health-check window
        "ContainerStartupHealthCheckTimeoutInSeconds": 900,
    }],
)

# 3. Endpoint
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)

# 4. Build a multipart body and invoke
body, content_type = build_multipart("audio.wav", {
    "language": "en",
    "response_format": "verbose_json",
    "diarize": "true",
})
resp = smrt.invoke_endpoint(EndpointName=NAME, ContentType=content_type, Body=body)
print(resp["Body"].read().decode())

# 5. Cleanup
sm.delete_endpoint(EndpointName=NAME)
sm.delete_endpoint_config(EndpointConfigName=NAME)
sm.delete_model(ModelName=NAME)
```

Real-time `InvokeEndpoint` is subject to SageMaker's **60-second response cap**. A cold first request can exceed it while a wav2vec2 aligner lazily
loads — retry, or use an [Asynchronous Endpoint](#asynchronous-endpoint) for long audio.

## Asynchronous Endpoint

Asynchronous inference removes the 60-second cap and is the recommended path for long audio. Reuse the `build_multipart` helper from the
[Real-Time Endpoint](#real-time-endpoint) example.

```python
import boto3
import json
import time

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")
s3 = boto3.client("s3")

REGION = boto3.session.Session().region_name
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = "public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023-sagemaker"
NAME = "whisperx-async"
# The AmazonSageMakerFullAccess role can only access buckets whose name contains "sagemaker"
BUCKET = f"sagemaker-{REGION}-{ACCOUNT}"

sm.create_model(
    ModelName=NAME,
    PrimaryContainer={"Image": IMAGE_URI},
    ExecutionRoleArn=ROLE_ARN,
)

sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g5.2xlarge",
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
        "ContainerStartupHealthCheckTimeoutInSeconds": 1200,
    }],
    AsyncInferenceConfig={
        "OutputConfig": {"S3OutputPath": f"s3://{BUCKET}/whisperx-async-output/"},
        # Match the container's single-worker limit
        "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
    },
)

sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)

# Upload the multipart body (audio embedded) to S3, then invoke by reference
body, content_type = build_multipart("audio.wav", {
    "response_format": "verbose_json",
    "diarize": "true",
})
input_key = "whisperx-async-input/request.bin"
s3.put_object(Bucket=BUCKET, Key=input_key, Body=body)

resp = smrt.invoke_endpoint_async(
    EndpointName=NAME,
    InputLocation=f"s3://{BUCKET}/{input_key}",
    ContentType=content_type,
)

# Poll the returned S3 output location for the JSON result
out_bucket, out_key = resp["OutputLocation"].replace("s3://", "").split("/", 1)
while True:
    try:
        result = s3.get_object(Bucket=out_bucket, Key=out_key)
        print(json.loads(result["Body"].read()))
        break
    except s3.exceptions.NoSuchKey:
        time.sleep(5)

# Cleanup
sm.delete_endpoint(EndpointName=NAME)
sm.delete_endpoint_config(EndpointConfigName=NAME)
sm.delete_model(ModelName=NAME)
```

Keep both the async input object and the `S3OutputPath` under a bucket whose name contains `sagemaker` (e.g. `sagemaker-<region>-<account>`); the
default execution-role policy (`AmazonSageMakerFullAccess`) grants S3 access only to such buckets.

## Notes

- **`InferenceAmiVersion` is required** on every GPU variant. WhisperX is a CUDA 12.8 image; the default SageMaker host AMI ships NVIDIA drivers that
  fail to start the container (a zero-log `CannotStartContainerError`). Pin `al2-ami-sagemaker-inference-gpu-3-1`.
- **One request per container.** Inference is serialized to a single transcription, so set `MaxConcurrentInvocationsPerInstance: 1` on async endpoints
  to match. Scale throughput by adding instances or containers, not concurrency. See [Known Limitations](../configuration.md#known-limitations).
- **Long audio** can exceed the 60-second real-time invoke cap — use an [Asynchronous Endpoint](#asynchronous-endpoint).

For all configuration options, see [Configuration](../configuration.md).
