# Supported Models

Each container serves a **single Whisper model**, pinned at launch through the `WHISPERX_DEFAULT_MODEL` environment variable (default `large-v2`).
Word-level alignment (wav2vec2) and speaker diarization (pyannote) run on top of that model's output. The request `model` field is ignored — to serve a
different model, launch another container. See [Configuration](../configuration.md) for all launch options.

## Whisper Models

`WHISPERX_DEFAULT_MODEL` accepts any faster-whisper model id. Weights download lazily from HuggingFace on first use (unless mounted locally — see
[Custom / BYO Models](#custom-byo-models)).

| Model | `WHISPERX_DEFAULT_MODEL` | Notes |
| --- | --- | --- |
| Tiny | `tiny` | Fastest, lowest accuracy |
| Base | `base` | |
| Small | `small` | |
| Medium | `medium` | Balanced speed and accuracy |
| Large v2 | `large-v2` | **Default** |
| Large v3 | `large-v3` | Latest large model |

All models are multilingual and support both `transcribe` and `translate` tasks (set via `WHISPERX_TASK`).

## Alignment and Diarization

On top of transcription, WhisperX adds two stages:

- **Word-level alignment (wav2vec2).** Requested per call with `timestamp_granularities[]=word` (or any subtitle output). A per-language wav2vec2
  phoneme model aligns each word to the audio. Aligners download lazily from HuggingFace per detected language and are cached with an LRU bound of
  `WHISPERX_ALIGN_LRU_SIZE` (default 3). Pin a specific aligner with `WHISPERX_ALIGN_MODEL`.
- **Speaker diarization (pyannote).** Requested per call with `diarize=true`, optionally bounded by `min_speakers` / `max_speakers`. Assigns a speaker
  label to each segment and word. The `speaker-diarization-community-1` pipeline is baked into the image, so diarization needs no network access.
  Diarization aligns words first, so it also requires the wav2vec2 aligner for the detected language.

## Custom / BYO Models

To serve a fine-tuned or private Whisper model, provide it as a **flat faster-whisper (CTranslate2) directory** — the CTranslate2 files must sit at the
root so faster-whisper loads the path directly:

```text
faster-whisper-model/
├── model.bin
├── config.json
├── tokenizer.json
├── vocabulary.json
└── preprocessor_config.json
```

The exact files vary by model and by the CTranslate2 converter version; what matters is that these files sit at the root of the directory (or tar archive), not in a nested subfolder.

- **{{ ec2_short }}** — mount the directory and point `WHISPERX_DEFAULT_MODEL` at the mount:

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/faster-whisper-model:/opt/ml/model:ro \
  -e WHISPERX_DEFAULT_MODEL=/opt/ml/model \
  public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023
```

- **{{ sagemaker }}** — package the same flat directory as a `model.tar.gz` (CTranslate2 files at the archive root) and pass it via `ModelDataUrl`.
  The entrypoint auto-detects it at `/opt/ml/model`. See [Specifying the Model](../deployment/sagemaker.md#specifying-the-model).

For fully offline use, see [Offline / Air-Gapped Usage](../deployment/ec2.md#offline-air-gapped-usage).

## Attribution

The container redistributes the [pyannote.audio](https://github.com/pyannote/pyannote-audio) `speaker-diarization-community-1` pipeline and its
`wespeaker-voxceleb-resnet34-LM` embedding model, unmodified, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit to Hervé Bredin,
Alexis Plaquet, the pyannote.audio contributors, and the WeSpeaker authors.
