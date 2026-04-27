"""Deploy a vLLM-Omni video model to a SageMaker async inference endpoint.

Video generation is async by design — /v1/videos returns a job ID immediately,
so only the job metadata JSON is written to S3, not the MP4 file. To retrieve
the MP4, poll /v1/videos/<id>/content directly against the endpoint.
"""

from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-sagemaker-v1.0.0",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    env={"SM_VLLM_MODEL": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g6e.xlarge",
    initial_instance_count=1,
    endpoint_name="vllm-omni-video-async",
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
    async_inference_config=AsyncInferenceConfig(
        output_path="s3://<BUCKET>/vllm-omni-async-output/",
        max_concurrent_invocations_per_instance=1,
    ),
    wait=True,
)

# The middleware converts the JSON payload to multipart/form-data for /v1/videos.
# Response contains the job ID; use the /v1/videos/<id>/content endpoint to
# retrieve the MP4 bytes directly from the container.
