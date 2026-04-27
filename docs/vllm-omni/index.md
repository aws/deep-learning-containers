# vLLM-Omni Inference

Pre-built Docker images for serving omni-modality models (text-to-speech, image generation, video generation, and multimodal chat) with
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). Built on Amazon Linux 2023 with CUDA 12.9 and Python 3.12.

## Latest Announcements

**vLLM-Omni 1.0.0** — Initial release. Serves TTS, image, video, and omni-chat models through OpenAI-compatible APIs. Includes a SageMaker routing
middleware for dispatching `/invocations` to any omni endpoint via `CustomAttributes`.

## Pull Commands

**EC2:**

```bash
docker pull {{ images.latest_vllm_omni_ec2 }}
```

**SageMaker:**

```bash
docker pull {{ images.latest_vllm_omni_sagemaker }}
```

See [Available Images](../reference/available_images.md) for all image URIs and [Getting Started](../get_started/index.md) for authentication
instructions.

## Packages

For package versions included in each release, see the [Release Notes](../releasenotes/vllm-omni/index.md).

## Supported Modalities

| Modality | Route | Example Model |
| --- | --- | --- |
| Text-to-Speech | `/v1/audio/speech` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Image Generation | `/v1/images/generations` | `black-forest-labs/FLUX.2-klein-4B` |
| Video Generation | `/v1/videos` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Multimodal Chat | `/v1/chat/completions` | `bytedance-research/BAGEL-7B-MoT`, `Qwen/Qwen2.5-Omni-3B` |

## Model Compatibility

- Models must have a standard HuggingFace `config.json` with a recognized `model_type`, or be diffusers pipeline models with `model_index.json`.
- Models requiring `--stage-configs-path` (e.g., CosyVoice3, Fish Speech) are not supported in v1.0.0 — the engine subprocess cannot resolve custom
  model types.
- Multi-stage omni models (thinker + talker + decoder) like Qwen2.5-Omni need significantly more VRAM than the model size suggests. Refer to the
  individual model cards for minimum GPU requirements.

## EC2 Deployment

The container runs `vllm serve --omni` and exposes the OpenAI-compatible API on port 8080. Each example below is a self-contained shell script that
starts the container, waits for readiness, submits a request, and writes the output to disk. Any `vllm serve` flag may be appended to `docker run`
(e.g., `--tensor-parallel-size 2`, `--max-model-len 2048`, `--enforce-eager`).

### Text-to-Speech

```bash
--8<-- "examples/vllm-omni/tts/run.sh"
```

### Image Generation

```bash
--8<-- "examples/vllm-omni/image/run.sh"
```

### Video Generation

The `/v1/videos` endpoint is asynchronous — it returns a job ID immediately and generates the video in the background. The script below submits the
job, polls until it completes, then downloads the MP4.

```bash
--8<-- "examples/vllm-omni/video/run.sh"
```

### Multimodal Chat

Use the standard OpenAI chat-completions API. Multimodal inputs (images, audio) are supplied as URL or base64 content parts in the message list.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 64}'
```

## SageMaker Deployment

### Prerequisites

- AWS CLI configured with appropriate permissions
- An IAM execution role with SageMaker and ECR permissions (see [Ray tutorial](../ray/index.md#prerequisites) for an example setup)
- SageMaker Python SDK v2:

```bash
pip install 'sagemaker>=2,<3'
```

### Routing Middleware

The SageMaker image includes an ASGI middleware that dispatches `/invocations` to the correct vllm-omni endpoint based on the `CustomAttributes`
header:

| `CustomAttributes` | Dispatched to |
| --- | --- |
| `route=/v1/audio/speech` | TTS |
| `route=/v1/images/generations` | Image generation |
| `route=/v1/videos` | Video generation (JSON auto-converted to form-data) |
| `route=/v1/chat/completions` | Multimodal chat |
| *(no route)* | vLLM default `/invocations` (chat/completion/embed) |

### Environment Variables

Any `SM_VLLM_*` env var is converted to a `--<name>` CLI argument (e.g., `SM_VLLM_MAX_MODEL_LEN=2048` → `--max-model-len 2048`).

| Variable | Description | Example |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID (HuggingFace or local path) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| `SM_VLLM_MAX_MODEL_LEN` | Max sequence length | `2048` |
| `SM_VLLM_ENFORCE_EAGER` | Disable CUDA graphs | `true` |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs for TP | `2` |
| `HF_TOKEN` | HuggingFace token for gated models | `hf_...` |

### Deploy a TTS Endpoint

!!! warning "SageMaker endpoint deployment takes several minutes and incurs costs. Remember to delete endpoints when done."

```python
--8<-- "examples/vllm-omni/sagemaker/deploy_tts.py"
```

GPU deploys require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 12.9 images. See
[ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid values.

When done, delete the endpoint:

```python
predictor.delete_endpoint()
```

### Async Inference for Video and Long-Running Generation

SageMaker real-time inference has a 60-second timeout. First requests to TTS models may exceed this due to `torch.compile` warmup (~67s); async
inference avoids the limit, as does retrying after warmup completes.

For `/v1/videos`, async inference is required because the endpoint returns a job ID rather than the final MP4. The MP4 must be retrieved by polling
the container directly — SageMaker async inference only captures the initial JSON response.

```python
--8<-- "examples/vllm-omni/sagemaker/deploy_video_async.py"
```

## Known Limitations

- **Video generation on SageMaker returns a job ID only.** The `/v1/videos` endpoint in v1.0.0 is async by design and `POST /v1/videos/sync` (which
  blocks and returns raw MP4 bytes) is not available. Direct container access (EC2) supports the full video workflow — create job, poll status,
  download MP4. A sync endpoint has been added in newer vllm-omni versions and will be supported in a future release.
- **First-request latency on SageMaker real-time endpoints.** TTS models can exceed the 60s invoke timeout on the first request due to `torch.compile`
  warmup. Use async inference or retry after warmup.

## Release Notes

See [vLLM-Omni Release Notes](../releasenotes/vllm-omni/index.md) for version history and changelogs.

## Resources

- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
