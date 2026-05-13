# vLLM-Omni Inference

Pre-built Docker images for serving omni-modality models (text-to-speech, audio generation, image generation, video generation, and multimodal chat)
with [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Built on Amazon Linux 2023 with CUDA 13.0 and Python 3.12.

## Latest Announcements

**May 12, 2026** — vLLM-Omni 0.20.0 release. Aligns with upstream vLLM v0.20.0; bumps CUDA to 13.0 and PyTorch to 2.11.0. Adds two new endpoints:
`/v1/audio/generate` for diffusion-based audio generation (e.g., stable-audio-open) and `/v1/videos/sync` — a blocking variant of `/v1/videos` that
returns the MP4 directly and unblocks video generation on SageMaker. New supported models: CosyVoice3, ERNIE-Image-Turbo, Wan2.1-VACE-1.3B,
Stable-Audio-Open-1.0.

**April 24, 2026** — vLLM-Omni 0.18.0 initial release. Serves TTS, image, video, and omni-chat models through OpenAI-compatible APIs. Includes a
SageMaker routing middleware for dispatching `/invocations` to any omni endpoint via `CustomAttributes`.

## Pull Commands

**EC2** — latest supported (floats across DLC minor versions; auto-upgrades on next pull):

```bash
docker pull {{ images.latest_vllm_omni_ec2 }}
```

**EC2** — patch-stable (recommended for production; auto-accepts DLC security patches in the v1.1 line, declines new DLC minor releases):

```bash
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-cuda-v1.1
```

**SageMaker** — latest supported:

```bash
docker pull {{ images.latest_vllm_omni_sagemaker }}
```

**SageMaker** — patch-stable:

```bash
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1.1
```

See [Available Images](../reference/available_images.md) for all image URIs, [Versioning and Tags](#versioning-and-tags) below for the convention, and
[Getting Started](../get_started/index.md) for authentication instructions.

## Versioning and Tags

vLLM-Omni image tags follow a **DLC-level** semantic versioning convention (independent of the bundled vllm-omni upstream version):

- **DLC major (`v1`, `v2`, …)** — incompatible/breaking changes in the DLC itself: image API, entrypoint, removed routes, pinned framework majors.
  Customer code may need updating when the DLC major bumps.
- **DLC minor (`v1.0`, `v1.1`, …)** — DLC release tracking new upstream vllm-omni features (e.g., a new endpoint), still API-compatible at the DLC
  level. May introduce behavioral changes in the bundled engine.
- **DLC patch** — security patches and bug fixes layered on top of an existing release without bumping the bundled vllm-omni version. Same tag, new
  image digest.

Two tag tiers, both floating, are exposed to customers:

- **Minor-floating tags** (`omni-cuda-v1`, `omni-sagemaker-cuda-v1`) — track the latest DLC release within a major line. Auto-upgrade across DLC minor
  *and* patch updates on `docker pull`. Best for development, quick-starts, and "give me whatever is supported right now".
- **Patch-floating tags** (`omni-cuda-v1.1`, `omni-sagemaker-cuda-v1.1`) — follow only the DLC patch stream within one minor release. They auto-accept
  security patches and bug fixes, but decline new DLC minor releases that could change behavior. Recommended for production: customers pinned here
  would have been insulated from the Code2Wav un-batching regression that landed with the DLC `v1.1` minor bump (see
  [Known Limitations](#known-limitations) below) until they were ready to evaluate it.

If your workload requires byte-identical reproducibility — i.e., declining even DLC patches — pull by digest instead of tag:

```bash
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm@sha256:<digest>
```

`docker inspect <image>` or `docker pull` output prints the digest of the image you currently have. Pulls by digest never change.

| Tag | Tracks | Currently points at |
| --- | --- | --- |
| `omni-cuda-v1` / `omni-sagemaker-cuda-v1` | latest DLC release in v1 line (minor + patch) | DLC `v1.1` (vllm-omni 0.20.0) |
| `omni-cuda-v1.0` / `omni-sagemaker-cuda-v1.0` | DLC v1.0 patch stream (vllm-omni 0.18.0 + DLC patches) | latest v1.0.x DLC patch |
| `omni-cuda-v1.1` / `omni-sagemaker-cuda-v1.1` | DLC v1.1 patch stream (vllm-omni 0.20.0 + DLC patches) | latest v1.1.x DLC patch |

## Packages

For package versions included in each release, see the [Release Notes](../releasenotes/vllm-omni/index.md).

## Supported Modalities

| Modality | Route | Example Models |
| --- | --- | --- |
| Text-to-Speech | `/v1/audio/speech` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `FunAudioLLM/CosyVoice3-0.5B` |
| Audio Generation | `/v1/audio/generate` (new in 0.20.0) | `stabilityai/stable-audio-open-1.0` |
| Image Generation | `/v1/images/generations` | `black-forest-labs/FLUX.2-klein-4B`, `baidu/ERNIE-Image-Turbo` |
| Video Generation (async) | `/v1/videos` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-VACE-1.3B-Diffusers` |
| Video Generation (sync) | `/v1/videos/sync` (new in 0.20.0) | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `Wan-AI/Wan2.1-VACE-1.3B-Diffusers` |
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

For voice cloning, use [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) or
[CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B) — both accept a reference audio clip plus its transcript and synthesize new
speech in the reference speaker's voice. CosyVoice3 is zero-shot voice-clone only (no preset voices) and requires `--trust-remote-code`.

```bash
--8<-- "examples/vllm-omni/tts/run.sh"
```

### Audio Generation

**Model:** [Stable-Audio-Open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) — a diffusion model for text-to-audio (sound effects,
ambience, short music clips), distinct from TTS. Generates up to ~47 seconds of audio per request, runs on a single 24 GB GPU.

The `/v1/audio/generate` endpoint (new in 0.20.0) takes a text prompt plus diffusion knobs (`audio_length`, `guidance_scale`, `num_inference_steps`,
`seed`) and returns a single binary WAV blob — no streaming. See the
[upstream API spec](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/audio_generate_api.md) for the full request shape.

```bash
--8<-- "examples/vllm-omni/audio-generate/run.sh"
```

### Image Generation

**Model:** [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — a 4B-parameter rectified-flow transformer from Black Forest
Labs, produces high-quality 512×512 images from text prompts, runs on a single 24 GB GPU.

[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) is also supported as of 0.20.0 — an 8-step distilled DiT for fast inference with a
matching request shape.

```bash
--8<-- "examples/vllm-omni/image/run.sh"
```

### Video Generation

**Model:** [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) — a 1.3B-parameter text-to-video diffusion model from the Wan
team, generates short clips at up to 480×832 resolution. Needs a 48 GB GPU (L40S) or 2× 24 GB GPUs with `--tensor-parallel-size 2`.
[Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-Diffusers) (added in 0.20.0) is a unified video creation/editing pipeline that
accepts text plus optional video, mask, or reference image inputs.

Two route options:

- **Async** (`POST /v1/videos`) — returns a job ID immediately; poll `GET /v1/videos/{id}` until status is `completed`, then download the MP4 from
  `GET /v1/videos/{id}/content`. Best for long-running batch jobs and the only option in 0.18.0.
- **Sync** (`POST /v1/videos/sync`, new in 0.20.0) — blocks until generation completes and returns the raw MP4 in the response body. Simpler client
  code, and crucially the only video path that works through SageMaker real-time endpoints (see [SageMaker Deployment](#sagemaker-deployment)).

```bash
--8<-- "examples/vllm-omni/video/run.sh"
```

```bash
--8<-- "examples/vllm-omni/video-sync/run.sh"
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
| `route=/v1/audio/generate` | Audio generation (new in 0.20.0) |
| `route=/v1/images/generations` | Image generation |
| `route=/v1/videos` | Video generation, async (JSON auto-converted to form-data) — returns job-ID only; MP4 not retrievable via SageMaker. Prefer `/v1/videos/sync` below. |
| `route=/v1/videos/sync` | Video generation, sync (new in 0.20.0) — blocks server-side and returns raw MP4 bytes; deploy behind SageMaker async inference (first-request `torch.compile` warmup exceeds the 60s real-time invoke timeout) |
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

GPU deploys require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 13.0 images. See
[ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid values.

When done, delete the endpoint:

```python
predictor.delete_endpoint()
```

### Async Inference for Long-Running TTS Generation

SageMaker real-time inference has a 60-second timeout. First requests to TTS models may exceed this due to `torch.compile` warmup (~67s); async
inference avoids the limit, as does retrying after warmup completes.

```python
--8<-- "examples/vllm-omni/sagemaker/deploy_tts_async.py"
```

For async inference, upload the JSON input payload to S3 first, then call `invoke_endpoint_async` with `InputLocation=<s3-uri>` and
`CustomAttributes="route=/v1/audio/speech"`. The resulting `.out` object in the configured S3 output path is the raw WAV audio — no polling or
additional retrieval step required.

### Deploy a Video Endpoint

The `/v1/videos/sync` endpoint (new in 0.20.0) is the supported path for video on SageMaker. Unlike the async `/v1/videos` route — which writes a
job-ID JSON to S3 but never the MP4 — `/v1/videos/sync` blocks server-side until generation completes and writes the raw MP4 bytes to the configured
S3 output path.

Deploy behind **SageMaker async inference** (`AsyncInferenceConfig`), not real-time inference: first-request latency on video models is dominated by
model load + `torch.compile` warmup (3–4 minutes for Wan2.1-VACE-1.3B), which exceeds the 60-second real-time invoke timeout. Async inference allows
up to 1 hour and writes the response body verbatim to S3, so the `.out` object *is* the MP4 — no polling on a job ID.

```python
--8<-- "examples/vllm-omni/sagemaker/deploy_video_sync.py"
```

Validated 2026-05-11 on `ml.g5.2xlarge` (A10G 24 GB VRAM, 32 GB host RAM): 45 KB MP4 in ~10s after warmup. Reduce `num_inference_steps` and
`num_frames` to stay under the async ceiling for warm requests.

## Known Limitations

- **`/v1/videos` (async) on SageMaker writes only the job-ID JSON to S3, not the MP4.** This is unchanged from 0.18.0 — the async route generates the
  MP4 in the background and the bytes never land in S3. Use the new `/v1/videos/sync` route on SageMaker (see
  [Deploy a Video Endpoint](#deploy-a-video-endpoint)) or stay on EC2 for the async workflow with status polling.
- **First-request latency on SageMaker real-time endpoints.** TTS, audio-generate, and video models can exceed the 60s invoke timeout on the first
  request due to `torch.compile` warmup. Use async inference or retry after warmup.
- **Voice-clone TTS (Qwen3-TTS-Base) is slower in 0.20.0 than 0.18.0 due to an upstream Code2Wav decode-chunk un-batching regression**
  ([vllm-omni#3203](https://github.com/vllm-project/vllm-omni/pull/3203)). Observed on `g6.xlarge` with `qwen3-tts-12hz-1.7b-base`, concurrency 4, 20
  prompts: requests/s **0.4 → 0.281**, audio RTF multiplier **1.6 → 1.109**, p95 E2E **11s → 15.9s**. TTS quality is unchanged. The fix is merged
  upstream as [vllm-omni#3485](https://github.com/vllm-project/vllm-omni/pull/3485) post-0.20.0 and will land in the next omni point release.
  Preset-voice TTS (Qwen3-TTS-CustomVoice) is unaffected.
- **CosyVoice3 requires `--trust-remote-code` and ~32 GB host RAM during model load.** A 16 GB host can SIGKILL the process during HuggingFace cache
  hydration. Prefer `g6e.xlarge` or larger for both EC2 and SageMaker instance types.
- **Stable-Audio-Open output is capped at ~47 seconds per request** by the model itself. For longer clips, run multiple requests with adjusted
  `audio_start` and concatenate client-side.

## Release Notes

See [vLLM-Omni Release Notes](../releasenotes/vllm-omni/index.md) for version history and changelogs.

## Resources

- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
