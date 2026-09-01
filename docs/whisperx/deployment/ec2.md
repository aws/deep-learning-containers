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

## WhisperX CLI (Batch Transcription)

The image also bundles the upstream [`whisperx`](https://github.com/m-bain/whisperX) command-line tool for one-off or batch file transcription. The
default entrypoint starts the HTTP server, so **override the entrypoint** to run the CLI instead. Mount an input and an output directory, and point
`--output_dir` (`-o`) at the mounted output — otherwise transcripts are written inside the container and lost when it exits.

The CLI's defaults differ from the server: `--model` defaults to `small` (pass `--model large-v2` to match the server) and `--output_format` defaults
to `all` (`srt`, `vtt`, `txt`, `tsv`, `json`, `aud`).

On a GPU host with a current NVIDIA driver:

```bash
docker run --rm --gpus all \
  -v "$PWD/audio:/audio:ro" \
  -v "$PWD/out:/out" \
  --entrypoint whisperx \
  public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023 \
  /audio/meeting.wav \
  --model large-v2 \
  --language en \
  --output_dir /out \
  --output_format srt
```

**With speaker diarization on GPU.** Point `--diarize_model` at the diarization pipeline baked into the image to diarize with no HuggingFace token or
network access. Overriding the entrypoint skips the automatic CUDA forward-compatibility step, so on GPU hosts whose NVIDIA driver predates the
image's CUDA 12.8 runtime, source the compat script first (a no-op on current drivers):

```bash
docker run --rm --gpus all \
  -v "$PWD/audio:/audio:ro" \
  -v "$PWD/out:/out" \
  --entrypoint bash \
  public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023 -lc '
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    source /opt/whisperx/start_cuda_compat.sh
    whisperx /audio/meeting.wav \
      --model large-v2 \
      --device cuda --compute_type float16 --batch_size 16 \
      --output_dir /out --output_format all \
      --diarize --diarize_model /opt/models/pyannote/speaker-diarization-community-1 \
      --min_speakers 1 --max_speakers 5
  '
```

- **Model downloads are ephemeral.** The Whisper model and wav2vec2 aligners download to the in-image cache (`HF_HOME=/opt/models/hf`) and are lost
  when the container exits. Mount a volume at `/opt/models/hf` to persist them across runs — do not mount over `/opt/models`, which would hide the
  baked diarization pipeline.
- **Default `--diarize` needs a token.** Without `--diarize_model`, the CLI downloads the gated `pyannote/speaker-diarization-community-1` model from
  HuggingFace and requires `--hf_token <token>`; the baked path above avoids both.
- Run `--entrypoint whisperx <image> --help` to list every flag.

## Configuration and Limits

- All launch options: [Configuration](../configuration.md).
- Inference is serialized to one request per container, uploads are capped, and `translate` cannot align or diarize — see
  [Known Limitations](../configuration.md#known-limitations).
