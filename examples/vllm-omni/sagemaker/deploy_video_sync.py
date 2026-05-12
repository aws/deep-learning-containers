"""Deploy a vLLM-Omni video model to a SageMaker real-time endpoint and invoke
the new /v1/videos/sync endpoint, which blocks until generation completes and
returns raw MP4 bytes.

Available since vLLM-Omni 0.20.0; supersedes the 0.18.0 limitation that
SageMaker async inference could only retrieve the job-ID JSON, not the MP4.

Use the routing middleware via `CustomAttributes="route=/v1/videos/sync"`,
which auto-converts JSON request bodies to multipart/form-data for the
underlying endpoint.
"""

import boto3
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    env={"SM_VLLM_MODEL": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.2xlarge",
    initial_instance_count=1,
    endpoint_name="vllm-omni-video-sync",
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
    wait=True,
)

# Invoke /v1/videos/sync via CustomAttributes; response body is the MP4 bytes
# (Content-Type: video/mp4). Prefer invoke_endpoint over invoke_endpoint_async
# because sync video can take 30–120s and the real-time path's binary response
# is what we want — async would write base64-encoded JSON to S3.
runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="vllm-omni-video-sync",
    Body='{"prompt": "a dog running on a beach", "num_frames": 17, '
    '"num_inference_steps": 30, "size": "480x320", "seed": 42}',
    ContentType="application/json",
    CustomAttributes="route=/v1/videos/sync",
)
with open("video.mp4", "wb") as f:
    f.write(response["Body"].read())

# When done:
# predictor.delete_endpoint()
