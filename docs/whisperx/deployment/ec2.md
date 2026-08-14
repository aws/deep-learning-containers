# EC2 Deployment

The container runs the WhisperX transcription API on port 8000. All model, decoding, VAD, and concurrency options are set at launch via `WHISPERX_*`
environment variables — the HTTP API takes an audio file, not tuning flags. See [Configuration](../configuration.md) for the full list.

## Single GPU

```bash
docker run -d --gpus all --shm-size=2g -p 8000:8000 \
  -e WHISPERX_DEFAULT_MODEL=large-v2 \
  public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023
```

On first boot the container warm-loads the Whisper model and diarization pipeline **before** binding the socket, so `/ping` refuses connections until
the models are resident (typically a few minutes). Wait for readiness, then transcribe an audio file (multipart upload, OpenAI-compatible):

```bash
until curl -sf http://localhost:8000/ping > /dev/null; do sleep 5; done

curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

Add `-F "language=en"` to skip auto-detection. `--shm-size=2g` is recommended for PyTorch shared-memory IPC. The entrypoint auto-activates CUDA
forward-compatibility when the host NVIDIA driver is older than the CUDA 12.8 runtime requires — no extra flag needed.

## Word Timestamps, Diarization, and Subtitles

**Word-level timestamps** — add the `word` granularity:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

**Speaker diarization** — set `diarize=true`, optionally bounding the speaker count:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

**Subtitles** — use `response_format=srt` or `vtt`:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt" \
  -F "max_line_width=42" \
  -F "highlight_words=true" \
  --output captions.srt
```

Requesting `srt`/`vtt` forces word-level alignment. The subtitle knobs (`max_line_width`, `max_line_count`, `highlight_words`) apply only to `srt` and
`vtt` output and are ignored for `json`, `text`, and `verbose_json`.

## Offline / Air-Gapped Usage

There is **no one-shot CLI or batch mode**. The entrypoint always starts the HTTP server, so `docker run <image> transcribe file.wav` is **not**
supported. To run without network access, pre-stage a model on the host, launch the server with the HuggingFace hub blocked, then POST to the local
endpoint.

1. Stage a flat faster-whisper (CTranslate2) model directory on the host — see [Custom / BYO Models](../models/index.md#custom-byo-models).
2. Launch with the model mounted and offline enforcement on:

```bash
docker run -d --gpus all --shm-size=2g -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e WHISPERX_DEFAULT_MODEL=/opt/ml/model \
  -v /path/to/local/model:/opt/ml/model:ro \
  public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023
```

3. Wait for readiness and transcribe against the local endpoint:

```bash
until curl -sf http://localhost:8000/ping > /dev/null; do sleep 5; done

curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=verbose_json"
```

**What works offline, and what to pre-cache:** The pyannote diarization pipeline is baked into the image and the VAD segmentation model ships inside
the WhisperX wheel, so both work with no network. Per-language **wav2vec2 aligners do not ship in the image** — they download lazily from HuggingFace
on the first word-timestamp **or diarize** request. With `HF_HUB_OFFLINE=1` and no cached aligner, those requests fail for an uncached language. So
fully-offline **segment-level transcription** works out of the box, but offline **word timestamps or diarization** require pre-caching the aligner
into `HF_HOME` (or pinning one with `WHISPERX_ALIGN_MODEL`) before going offline.

## Configuration and Limits

- All launch options: [Configuration](../configuration.md).
- Inference is serialized to one request per container, uploads are capped, and `translate` cannot align or diarize — see
  [Known Limitations](../configuration.md#known-limitations).
