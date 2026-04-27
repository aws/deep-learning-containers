"""Deploy a vLLM-Omni TTS model to a real-time SageMaker endpoint."""

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
    endpoint_name="vllm-omni-tts",
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
    wait=True,
)

# Invoke — route /invocations to /v1/audio/speech via CustomAttributes
sm_runtime = predictor.sagemaker_session.sagemaker_runtime_client
response = sm_runtime.invoke_endpoint(
    EndpointName=predictor.endpoint_name,
    ContentType="application/json",
    Body='{"input": "Hello world", "voice": "vivian", "language": "English"}',
    CustomAttributes="route=/v1/audio/speech",
)
with open("speech.wav", "wb") as f:
    f.write(response["Body"].read())
