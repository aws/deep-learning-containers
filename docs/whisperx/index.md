# Speech Transcription using WhisperX DLC

Production-ready Docker images for transcribing, aligning, and diarizing speech with [WhisperX](https://github.com/m-bain/whisperX) on {{ aws }}.
Built on Amazon Linux 2023 with ongoing security patching.

Transcribe audio to text, align words to precise timestamps, label speakers, and export subtitles through an OpenAI-compatible API.

## Images

| Platform | Image | Default Port |
| --- | --- | --- |
| {{ ec2_short }} | `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023` | 8000 |
| {{ sagemaker }} | `public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023-sagemaker` | 8080 |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/whisperx). For private ECR URIs, see
[Image Access](../get_started/index.md).

## What's Included

In addition to WhisperX and its core stack (PyTorch 2.8, CUDA 12.8, Python 3.12), the images bundle:

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1** — Whisper inference engine backed by CTranslate2
- **[CTranslate2](https://github.com/OpenNMT/CTranslate2) 4.8.0** — the optimized runtime that executes the Whisper models
- **wav2vec2 forced alignment** — per-word timestamps from phoneme-level alignment models, applied on top of the transcript
- **[pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.0.7** — speaker diarization pipeline, baked into the image for offline use
- **ffmpeg** — audio decoding for any input container or codec
- **[FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)** — the OpenAI-compatible HTTP server

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/transcriptions` | Transcribe an audio file (OpenAI-compatible) |
| `POST /invocations` | {{ sm_short }} alias — identical behavior to `/v1/audio/transcriptions` |
| `GET /ping` | Readiness health check |
| `GET /v1/models` | Advertise the served model id |

The request is a `multipart/form-data` file upload, not a JSON body. See [EC2 Deployment](deployment/ec2.md) and
[{{ sagemaker }} Deployment](deployment/sagemaker.md) for examples, and [Configuration](configuration.md) for every launch option and request field.

## How We Build

These images are curated builds tracking the [WhisperX](https://github.com/m-bain/whisperX) project:

- **Built from upstream releases** — images track WhisperX releases, each gated by our regression test suite before publication.
- **Regression-tested** — validated for transcription, word alignment, and diarization on {{ ec2_short }} and {{ sagemaker }} on every release. See
  [Supported Models](models/index.md).
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
