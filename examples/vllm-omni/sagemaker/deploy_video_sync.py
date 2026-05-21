"""Deploy a vLLM-Omni video model to a SageMaker async inference endpoint
and invoke the new /v1/videos/sync endpoint, which blocks server-side until
generation completes and returns raw MP4 bytes.

Async inference is required for video — first-request latency includes model
load + torch.compile warmup (3-4 min for Wan2.1-VACE-1.3B), well past the
60s real-time invoke timeout. Async inference allows up to 1 hour and
deposits the response body verbatim at the configured S3 output path, so the
.out object is the raw MP4.

Available since vLLM-Omni 0.20.0; supersedes the 0.18.0 limitation that
SageMaker async inference could only retrieve the job-ID JSON, not the MP4.
/v1/videos/sync expects multipart/form-data input; SageMaker InvokeEndpoint
forwards the request body and ContentType through to the model server
unchanged, so the client builds the multipart body locally and uploads it
with ContentType="multipart/form-data; boundary=...".

Validated 2026-05-11 on ml.g5.2xlarge (A10G 24 GB VRAM, 32 GB host RAM):
45 KB MP4 returned in ~10s after warmup.
"""

import time
import uuid

import boto3
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor

BUCKET = "<BUCKET>"  # replace with an S3 bucket your role can read/write
ROLE_ARN = "arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole"
ENDPOINT_NAME = "vllm-omni-video-sync"

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1",
    role=ROLE_ARN,
    env={"SM_VLLM_MODEL": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    endpoint_name=ENDPOINT_NAME,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    async_inference_config=AsyncInferenceConfig(
        output_path=f"s3://{BUCKET}/vllm-omni-async-output/",
        max_concurrent_invocations_per_instance=1,
    ),
    wait=True,
)


def build_multipart_body(fields: dict, boundary: str) -> bytes:
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        for k, v in fields.items()
    ]
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()


# Build a multipart/form-data body locally; upload to S3, then invoke async.
boundary = uuid.uuid4().hex
content_type = f"multipart/form-data; boundary={boundary}"
body = build_multipart_body(
    {
        "prompt": "a dog running on a beach",
        "num_frames": "17",
        "num_inference_steps": "4",
        "size": "480x320",
        "seed": "42",
    },
    boundary,
)

s3 = boto3.client("s3")
s3.put_object(
    Bucket=BUCKET,
    Key="vllm-omni-async-input/request.bin",
    Body=body,
    ContentType=content_type,
)

runtime = boto3.client("sagemaker-runtime")
result = runtime.invoke_endpoint_async(
    EndpointName=ENDPOINT_NAME,
    InputLocation=f"s3://{BUCKET}/vllm-omni-async-input/request.bin",
    ContentType=content_type,
    CustomAttributes="route=/v1/videos/sync",
)
output_location = result["OutputLocation"]  # s3://.../<id>.out
print(f"Output will be written to {output_location}")

# Poll for the .out object (raw MP4 bytes). First request takes ~3-4 min
# due to model load + torch.compile; warm requests are ~3-10s.
bucket = output_location.split("/", 3)[2]
key = output_location.split("/", 3)[3]
for _ in range(120):  # 10 min timeout
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        with open("video.mp4", "wb") as f:
            f.write(obj["Body"].read())
        print(f"wrote video.mp4 (Content-Type: {obj.get('ContentType', '?')})")
        break
    except s3.exceptions.NoSuchKey:
        time.sleep(5)
else:
    raise RuntimeError("timed out waiting for async output")

# When done:
# predictor.delete_endpoint()
