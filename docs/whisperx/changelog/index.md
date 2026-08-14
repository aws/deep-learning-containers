# Changelog

Changelog for the Amazon Linux 2023-based WhisperX images (`3.8.6-cu128-amzn2023`, `3.8.6-cu128-amzn2023-sagemaker`).

* * *

## v1.0.0 — 2026-08-14

<!-- Release date is provisional; confirm at publish. -->

**Tags:** `3.8.6-cu128-amzn2023` · `3.8.6-cu128-amzn2023-sagemaker`

**WhisperX source:** [v3.8.6](https://github.com/m-bain/whisperX/releases/tag/v3.8.6)

### Highlights

- Initial release of WhisperX containers on Amazon Linux 2023.
- Speech transcription, word-level alignment (wav2vec2), and speaker diarization (pyannote) through an OpenAI-compatible API.
- Deployable on {{ ec2_short }} (port 8000) and {{ sagemaker }} (port 8080, real-time and asynchronous endpoints).
- Built on CUDA 12.8 with Python 3.12; faster-whisper 1.2.1, CTranslate2 4.8.0, and pyannote.audio 4.0.7.

### Supported Models at Launch

- Whisper `tiny`, `base`, `small`, `medium`, `large-v2` (default), and `large-v3` — served one per container via `WHISPERX_DEFAULT_MODEL`.
- Per-language wav2vec2 alignment models, downloaded on demand.
- pyannote `speaker-diarization-community-1` diarization pipeline, baked into the image.
