# Configuration

All tuning is set at container launch via `WHISPERX_*` environment variables. There are no per-request tuning fields — the HTTP API takes an audio
file plus a small set of output-shaping form fields (see [Request Fields](#request-fields)). These variables are read once at startup into immutable
settings; pass them with `-e` on {{ ec2_short }} or via the container `Environment` on {{ sagemaker }}.

Boolean variables accept `1`, `true`, `yes`, or `on` (case-insensitive) as true; anything else is false.

## Model and Pipeline

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPERX_DEFAULT_MODEL` | `large-v2` | Whisper model served by the container — a faster-whisper model id (`tiny`, `large-v3`, …) or a local directory path |
| `WHISPERX_SERVED_MODEL_NAME` | *(unset)* | Client-facing alias advertised by `GET /v1/models` (e.g. `whisper-1` for OpenAI-SDK drop-in). Does not change inference |
| `WHISPERX_COMPUTE_TYPE` | `float16` | CTranslate2 compute type |
| `WHISPERX_BATCH_SIZE` | `16` | faster-whisper batch size |
| `WHISPERX_VAD_METHOD` | `pyannote` | Voice-activity-detection backend |
| `WHISPERX_TASK` | `transcribe` | `transcribe` or `translate` (translate outputs English and cannot word-align — see [Known Limitations](#known-limitations)) |
| `WHISPERX_ALIGN_MODEL` | *(unset)* | Pin a specific wav2vec2 aligner; unset uses the WhisperX default per language |
| `WHISPERX_DIARIZE_MODEL_PATH` | `/opt/models/pyannote/speaker-diarization-community-1` | Baked pyannote diarization pipeline directory |
| `WHISPERX_ALIGN_LRU_SIZE` | `3` | Maximum resident wav2vec2 aligners (keyed by language) |

## Decoding

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPERX_TEMPERATURE` | `0.0` | Sampling temperature |
| `WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK` | `0.2` | Temperature step applied on decode fallback |
| `WHISPERX_BEAM_SIZE` | `5` | Beam-search width |
| `WHISPERX_BEST_OF` | `5` | Number of candidates when sampling |
| `WHISPERX_PATIENCE` | `1.0` | Beam-search patience |
| `WHISPERX_LENGTH_PENALTY` | `1.0` | Length penalty |
| `WHISPERX_COMPRESSION_RATIO_THRESHOLD` | `2.4` | Reject a decode whose gzip compression ratio exceeds this |
| `WHISPERX_LOGPROB_THRESHOLD` | `-1.0` | Reject a decode whose average log-probability is below this |
| `WHISPERX_NO_SPEECH_THRESHOLD` | `0.6` | Treat a segment as silence above this no-speech probability |
| `WHISPERX_CONDITION_ON_PREVIOUS_TEXT` | `false` | Feed previously decoded text as context for the next window |
| `WHISPERX_INITIAL_PROMPT` | *(unset)* | Optional text prompt for the first window |
| `WHISPERX_HOTWORDS` | *(unset)* | Hotword / bias phrases |
| `WHISPERX_SUPPRESS_TOKENS` | `-1` | Comma-separated token ids to suppress |
| `WHISPERX_SUPPRESS_NUMERALS` | `false` | Suppress numeric tokens |

## Voice Activity Detection (VAD)

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPERX_CHUNK_SIZE` | `30` | VAD chunk size (seconds) |
| `WHISPERX_VAD_ONSET` | `0.5` | VAD onset threshold |
| `WHISPERX_VAD_OFFSET` | `0.363` | VAD offset threshold |

## Concurrency and Limits

| Variable | Default | Description |
| --- | --- | --- |
| `WHISPERX_MAX_CONCURRENT_REQUESTS` | `1` | Concurrent transcriptions — **clamped to 1**; any other value is ignored with a warning |
| `WHISPERX_MAX_QUEUE` | `2` | Requests allowed to wait while one runs; excess is shed with HTTP 503 |
| `WHISPERX_MAX_UPLOAD_BYTES` | `104857600` | Maximum upload size in bytes (100 MiB); larger uploads return HTTP 413 |

## Model Cache and Offline

| Variable | Default | Description |
| --- | --- | --- |
| `HF_HOME` | `/opt/models/hf` | HuggingFace cache (Whisper weights and wav2vec2 aligners). On {{ sagemaker }}, repointed to `/opt/ml/model` when that directory is populated |
| `TORCH_HOME` | `/opt/models/torch` | Torch hub cache |
| `HF_HUB_OFFLINE` | `0` | Set `1` to block all HuggingFace network access |
| `TRANSFORMERS_OFFLINE` | `0` | Set `1` to block Transformers network access |

To run without network access, stage a model locally and set these to `1` — see [Custom / BYO Models](models/index.md#custom-byo-models).

## Request Fields

Form fields on `POST /v1/audio/transcriptions` and `POST /invocations` (identical on both):

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | *(required)* | The audio file (multipart upload) |
| `language` | string | auto-detect | ISO code (e.g. `en`, `zh`); omit to auto-detect |
| `response_format` | string | `json` | One of `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities[]` | list | `segment` | Add `word` for word-level timestamps |
| `diarize` | bool | `false` | Assign speaker labels (WhisperX extension) |
| `min_speakers` | int | *(unset)* | Lower bound on speaker count (≥ 1) |
| `max_speakers` | int | *(unset)* | Upper bound on speaker count (≥ `min_speakers`) |
| `max_line_width` | int | *(unset)* | `srt`/`vtt` only — maximum characters per line |
| `max_line_count` | int | *(unset)* | `srt`/`vtt` only — maximum lines per cue |
| `highlight_words` | bool | `false` | `srt`/`vtt` only — highlight each word as it is spoken |

## Known Limitations

- **Inference is serialized to one request per container.** A single WhisperX pipeline is not concurrency-safe, so `WHISPERX_MAX_CONCURRENT_REQUESTS`
  is clamped to 1. Up to `WHISPERX_MAX_QUEUE` (default 2) requests queue while one runs; further requests are shed immediately with HTTP 503
  `"server busy: inference queue full"`. The default in-flight ceiling is **1 running + 2 queued = 3**. Scale throughput by running **more
  containers**, not by raising concurrency.
- **Upload size is capped** at `WHISPERX_MAX_UPLOAD_BYTES` (default 100 MiB); larger uploads are rejected with HTTP 413.
- **`task=translate` cannot word-align or diarize.** Translated English text cannot be aligned to the source-language audio; a `diarize=true` request
  with `WHISPERX_TASK=translate` returns HTTP 422.
- **Long audio on SageMaker real-time endpoints** can exceed the 60-second invoke timeout. Use
  [SageMaker async inference](deployment/sagemaker.md#asynchronous-endpoint) for long files.

## Full Reference

- [WhisperX](https://github.com/m-bain/whisperX)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
