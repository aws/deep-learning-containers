# vLLM-Omni Inference

Pre-built Docker images for serving omni-modality models (text-to-speech, image generation, video generation, and multimodal chat) with
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). Built on Amazon Linux 2023 with CUDA 12.9 and Python 3.12.

## Latest Announcements

**April 24, 2026** — vLLM-Omni 0.18.0 initial release. Serves TTS, image, video, and omni-chat models through OpenAI-compatible APIs. Includes a
SageMaker routing middleware for dispatching `/invocations` to any omni endpoint via `CustomAttributes`.

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
- Some HuggingFace repos ship a `config.json` without a `model_type` field; vllm-omni's config resolver will reject these. Patching the local snapshot
  with a minimal `config.json` (`{"model_type": "...", "architectures": ["..."]}`) is a common workaround, but the container's pinned `transformers`
  version must also register the model type — models newer than that pin will fail at engine startup. Upgrading `transformers` in-place risks breaking
  the supported models; wait for a future vllm-omni release with an updated pin.
- Multi-stage omni models (thinker + talker + decoder) like Qwen2.5-Omni need significantly more VRAM than the model size suggests. Refer to the
  individual model cards for minimum GPU requirements.

## EC2 Deployment

The container runs `vllm serve --omni` and exposes the OpenAI-compatible API on port 8080. Each example below is a self-contained shell script that
starts the container, waits for readiness, submits a request, and writes the output to disk. Any `vllm serve` flag may be appended to `docker run`
(e.g., `--tensor-parallel-size 2`, `--max-model-len 2048`, `--enforce-eager`).

### Text-to-Speech

**Model:** [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — a 1.7B-parameter Qwen3 text-to-speech
model supporting multiple voices and languages, runs on a single 24 GB GPU (A10G / L4).

```bash
--8<-- "examples/vllm-omni/tts/run.sh"
```

### Image Generation

**Model:** [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — a 4B-parameter rectified-flow transformer from Black Forest
Labs, produces high-quality 512×512 images from text prompts, runs on a single 24 GB GPU.

```bash
--8<-- "examples/vllm-omni/image/run.sh"
```

### Video Generation

**Model:** [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) — a 1.3B-parameter text-to-video diffusion model from the Wan
team, generates short clips at up to 480×832 resolution. Needs a 48 GB GPU (L40S) or 2× 24 GB GPUs with `--tensor-parallel-size 2`.

The `/v1/videos` endpoint is asynchronous — it returns a job ID immediately and generates the video in the background. The script below submits the
job, polls until it completes, then downloads the MP4.

```bash
--8<-- "examples/vllm-omni/video/run.sh"
```

### Multimodal Chat

Use the standard OpenAI chat-completions API. Multimodal inputs (images, audio) are supplied as URL or base64 content parts in the message list.

**Example model:** [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) — a 3B-parameter omni model accepting text, image, and audio inputs
and generating text or speech outputs. Multi-stage architecture (thinker + talker + code2wav) requires **≥ 4 GPUs**: `g5.12xlarge` / `g6.12xlarge` (4×
A10G) or `g6e.12xlarge` (4× L40S).

Start the server, then submit a request. Three things are **required** on `/v1/chat/completions` to produce clean audio from Qwen2.5-Omni:

1. `"modalities": ["audio"]` — not `["text","audio"]` (that returns empty audio).
2. `"sampling_params_list"` — a 3-element list (thinker, talker, code2wav). The image's built-in per-stage defaults produce noise; use the values from
   the official Qwen docs.
3. The exact Qwen system prompt.

!!! warning "Omitting `sampling_params_list` returns 200 with valid WAV bytes that sound like noise — the single most common footgun."

```bash
--8<-- "examples/vllm-omni/qwen2.5-omni/run.sh"
```

The `/v1/audio/speech` shortcut (voices: `Chelsie`, `Ethan`) bypasses the thinker and does not apply the correct sampling params in 0.18.0, so it
produces noisy output for Qwen2.5-Omni. Prefer `/v1/chat/completions` for this model.

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
| `route=/v1/videos` | Video generation (JSON auto-converted to form-data) — returns job-ID only in 0.18.0, MP4 not retrievable via SageMaker |
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

### Async Inference for Long-Running TTS Generation

SageMaker real-time inference has a 60-second timeout. First requests to TTS models may exceed this due to `torch.compile` warmup (~67s); async
inference avoids the limit, as does retrying after warmup completes.

!!! warning "Video generation is not supported on SageMaker in 0.18.0 — see [Known Limitations](#known-limitations) below. Use EC2 for video."

```python
--8<-- "examples/vllm-omni/sagemaker/deploy_tts_async.py"
```

For async inference, upload the JSON input payload to S3 first, then call `invoke_endpoint_async` with `InputLocation=<s3-uri>` and
`CustomAttributes="route=/v1/audio/speech"`. The resulting `.out` object in the configured S3 output path is the raw WAV audio — no polling or
additional retrieval step required.

## Known Limitations

- **Video generation is not supported on SageMaker in 0.18.0.** The `/v1/videos` endpoint is async by design — it returns a job-ID JSON immediately
  and generates the MP4 in the background. Through SageMaker async inference, only that job-ID JSON is written to S3; the MP4 itself never lands in S3
  and cannot be retrieved through `invoke_endpoint` or `invoke_endpoint_async`. Use EC2 for video generation — direct container access supports the
  full workflow (create job, poll status, download MP4). SageMaker support is expected once `POST /v1/videos/sync` (which blocks and returns raw MP4
  bytes) is available in a future vllm-omni release.
- **First-request latency on SageMaker real-time endpoints.** TTS models can exceed the 60s invoke timeout on the first request due to `torch.compile`
  warmup. Use async inference or retry after warmup.

## Release Notes

See [vLLM-Omni Release Notes](../releasenotes/vllm-omni/index.md) for version history and changelogs.

## Resources

- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
