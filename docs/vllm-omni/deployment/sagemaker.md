# Amazon SageMaker AI Deployment

The SageMaker image (`omni-sagemaker-cuda`) includes a routing middleware that dispatches `/invocations` to the correct vLLM-Omni endpoint based on
the `CustomAttributes` header.

## Routing

| `CustomAttributes` | Dispatched to |
| --- | --- |
| `route=/v1/audio/speech` | TTS |
| `route=/v1/audio/generate` | Audio generation |
| `route=/v1/images/generations` | Image generation |
| `route=/v1/videos/sync` | Video generation (sync, returns MP4) |
| `route=/v1/chat/completions` | Multimodal chat |
| *(no route)* | vLLM default `/invocations` |

## Deploy a TTS Endpoint

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

model = Model(
    image_uri="public.ecr.aws/deep-learning-containers/vllm:omni-sagemaker-cuda",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
    env={"SM_VLLM_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
)

response = predictor.predict(
    data={"input": "Hello, how are you?", "voice": "vivian", "language": "English"},
    custom_attributes="route=/v1/audio/speech",
)

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 13.0 images.
- First requests to TTS, audio, and video models may exceed the 60-second real-time invoke timeout due to `torch.compile` warmup. Use SageMaker async
  inference for long-running generation tasks.
- The `/v1/videos` async route writes only the job-ID JSON to S3, not the MP4. Use `/v1/videos/sync` for video generation on SageMaker.

For all configuration options, see [Configuration](../configuration.md).
