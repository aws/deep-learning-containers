"""Deploy a vLLM-Omni TTS model to a SageMaker async inference endpoint.

Async inference avoids the 60-second real-time invoke timeout, which the first
TTS request can exceed due to torch.compile warmup (~67s). The /v1/audio/speech
endpoint returns raw WAV bytes, so the async output written to S3 is the usable
audio file — no polling or extra retrieval step needed.
"""

from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    env={"SM_VLLM_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="vllm-omni-tts-async",
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
    async_inference_config=AsyncInferenceConfig(
        output_path="s3://<BUCKET>/vllm-omni-async-output/",
        max_concurrent_invocations_per_instance=1,
    ),
    wait=True,
)

# Invoke async — upload the JSON input to S3, then call invoke_endpoint_async.
# The resulting .out object in S3 is the raw WAV audio bytes (content-type audio/wav).
# Use CustomAttributes to route /invocations → /v1/audio/speech.
