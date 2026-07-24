# Amazon SageMaker AI Deployment

The SageMaker image (`omni-sagemaker-cuda`) includes a routing middleware that dispatches `/invocations` to the correct vLLM-Omni endpoint based on
the `CustomAttributes` header.

## Specifying the Model

The SageMaker image resolves the model in this order:

1. **`SM_VLLM_MODEL` environment variable** — explicit Hugging Face ID or path
2. **`/opt/ml/model`** — when SageMaker mounts model artifacts via `ModelDataUrl` or `ModelDataSource`, the entrypoint auto-detects them
3. **`HF_MODEL_ID` environment variable** — fallback Hugging Face ID

Set exactly one. For gated models, also pass `HF_TOKEN`.

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

## Other Modalities

The same pattern works for image and sync-video endpoints — change `SM_VLLM_MODEL`, `instance_type`, and the `route=` attribute.

```python
# Image generation — FLUX.2-klein-4B on g6.xlarge (JSON)
response = predictor.predict(
    data={"prompt": "a red apple on a white table", "size": "512x512", "n": 1},
    custom_attributes="route=/v1/images/generations",
)
```

The `/v1/videos` and `/v1/videos/sync` routes accept either `application/json` or `multipart/form-data`. As of `omni-sagemaker-cuda-v1.4`, the routing
middleware again converts a JSON body to multipart for these routes (this was disabled in `v1.2` and restored in `v1.4`), so sending JSON — as in the
image and chat examples above — works. The `multipart/form-data` path below is still supported unchanged and remains useful when you want
byte-for-byte control over the request body.

```python
import uuid
import boto3

def build_multipart_body(fields: dict, boundary: str) -> bytes:
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        for k, v in fields.items()
    ]
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()

# Sync video — Wan2.1-T2V-1.3B on g6e.xlarge
boundary = uuid.uuid4().hex
content_type = f"multipart/form-data; boundary={boundary}"
body = build_multipart_body(
    {"prompt": "a dog running on a beach", "num_frames": "17",
     "num_inference_steps": "30", "size": "480x320"},
    boundary,
)

runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName=predictor.endpoint_name,
    Body=body,
    ContentType=content_type,
    CustomAttributes="route=/v1/videos/sync",
)
mp4_bytes = response["Body"].read()
```

For the async `/v1/videos` route on SageMaker async inference, see
[examples/vllm-omni/sagemaker/deploy_video_sync.py](https://github.com/aws/deep-learning-containers/blob/main/examples/vllm-omni/sagemaker/deploy_video_sync.py).

For Qwen2.5-Omni multimodal chat (text + speech output), set `route=/v1/chat/completions` and use the request shape from the
[EC2 multimodal-chat example](ec2.md#speech-output-requirements).

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 13.0 images.
- First requests to TTS, audio, and video models may exceed the 60-second real-time invoke timeout due to `torch.compile` warmup. Use SageMaker async
  inference for long-running generation tasks.
- The `/v1/videos` async route writes only the job-ID JSON to S3, not the MP4. Use `/v1/videos/sync` for video generation on SageMaker.

For all configuration options, see [Configuration](../configuration.md).
