"""WhisperX FastAPI server.

Design ref: workspace/whisperx-docker/DESIGN.md §5.

Four routes, all served in one process:
    POST /v1/audio/transcriptions   — primary, OpenAI-compatible
    POST /invocations               — alias for SageMaker
    GET  /ping                      — shallow health check
    GET  /v1/models                 — list Whisper sizes cached in HF_HOME

Extension fields on top of OpenAI's schema: `diarize`, `min_speakers`,
`max_speakers`. When `diarize=false` (default) output is byte-identical to
OpenAI's `verbose_json`.
"""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from whisperx.diarize import DiarizationPipeline

# ---------------------------------------------------------------------------
# Model LRUs
# ---------------------------------------------------------------------------
# Whisper transcription models — keyed by name. Users pick these per-request
# via the OpenAI-compatible `model` field.
_WHISPER_LRU: "OrderedDict[str, Any]" = OrderedDict()
_WHISPER_LRU_MAX = int(os.environ.get("WHISPERX_MODEL_LRU_SIZE", "2"))

# wav2vec2 align models — keyed by language code.
_ALIGN_LRU: "OrderedDict[str, tuple[Any, dict[str, Any]]]" = OrderedDict()
_ALIGN_LRU_MAX = int(os.environ.get("WHISPERX_ALIGN_LRU_SIZE", "3"))

# Fixed models loaded at startup.
_DIARIZE_PIPELINE: Any = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# The Whisper model pinned at container launch and warmed in the lifespan hook.
DEFAULT_MODEL = os.environ.get("WHISPERX_DEFAULT_MODEL", "large-v2")
# Optional client-facing alias for the pinned model (e.g. "whisper-1" for an
# OpenAI-SDK drop-in). Requests naming this alias resolve to DEFAULT_MODEL.
SERVED_MODEL_NAME = os.environ.get("WHISPERX_SERVED_MODEL_NAME")
# When truthy, requests may name any Whisper model and it is loaded on demand
# via _WHISPER_LRU (the pre-pinning behavior). Otherwise the `model` field is
# validated against the served id and mismatches are rejected.
ALLOW_MODEL_OVERRIDE = os.environ.get("WHISPERX_ALLOW_MODEL_OVERRIDE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
DIARIZE_MODEL_PATH = os.environ.get(
    "WHISPERX_DIARIZE_MODEL_PATH",
    "/opt/models/pyannote/speaker-diarization-community-1",
)


def _lru_touch(cache: "OrderedDict[str, Any]", key: str, max_size: int) -> None:
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _get_whisper(model_name: str) -> Any:
    if model_name in _WHISPER_LRU:
        _lru_touch(_WHISPER_LRU, model_name, _WHISPER_LRU_MAX)
        return _WHISPER_LRU[model_name]
    model = whisperx.load_model(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
    _WHISPER_LRU[model_name] = model
    _lru_touch(_WHISPER_LRU, model_name, _WHISPER_LRU_MAX)
    return model


def _get_align(language: str) -> tuple[Any, dict[str, Any]]:
    if language in _ALIGN_LRU:
        _lru_touch(_ALIGN_LRU, language, _ALIGN_LRU_MAX)
        return _ALIGN_LRU[language]
    model, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
    _ALIGN_LRU[language] = (model, metadata)
    _lru_touch(_ALIGN_LRU, language, _ALIGN_LRU_MAX)
    return _ALIGN_LRU[language]


def _resolve_model(requested: str | None) -> str:
    """Map a request's `model` field to the concrete Whisper model to run.

    The Whisper model is pinned at launch (DEFAULT_MODEL). A request may omit
    `model`, name the pinned model, or name its SERVED_MODEL_NAME alias — all
    resolve to DEFAULT_MODEL. Any other value is rejected unless
    ALLOW_MODEL_OVERRIDE is set, which restores per-request model loading.
    """
    if not requested:
        return DEFAULT_MODEL
    if requested == DEFAULT_MODEL:
        return DEFAULT_MODEL
    if SERVED_MODEL_NAME and requested == SERVED_MODEL_NAME:
        return DEFAULT_MODEL
    if ALLOW_MODEL_OVERRIDE:
        return requested
    raise HTTPException(
        status_code=404,
        detail=f"The model `{requested}` does not exist. This endpoint serves `{SERVED_MODEL_NAME or DEFAULT_MODEL}`.",
    )


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load fixed models before uvicorn binds so /ping stays shallow but honest.

    Design §6 decision 8: `/ping` reachable ⇒ pyannote + Silero are resident.
    """
    global _DIARIZE_PIPELINE
    # Warm the default Whisper model so the first request isn't a cold ~30 s
    # download-and-load.
    _get_whisper(DEFAULT_MODEL)

    # Diarization: load the pyannote pipeline from the fixed local path the
    # image baked at build time (COPYed from S3). Explicit local path ⇒ no HF
    # token and no network needed. Absence is not fatal — non-diarized requests
    # still work. This path is independent of HF_HOME, so the SageMaker
    # entrypoint repointing HF_HOME (for user-mounted Whisper models) does not
    # hide it.
    if os.path.isdir(DIARIZE_MODEL_PATH):
        try:
            _DIARIZE_PIPELINE = DiarizationPipeline(
                model_name=DIARIZE_MODEL_PATH,
                device=DEVICE,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort startup
            print(f"WARN: diarization pipeline failed to load from {DIARIZE_MODEL_PATH}: {exc}")
            _DIARIZE_PIPELINE = None
    else:
        print(f"WARN: diarization model dir {DIARIZE_MODEL_PATH} missing; diarization disabled")
        _DIARIZE_PIPELINE = None

    yield


app = FastAPI(title="WhisperX DLC", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _seconds_to_srt_ts(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def _seconds_to_vtt_ts(t: float) -> str:
    return _seconds_to_srt_ts(t).replace(",", ".")


def _to_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_seconds_to_srt_ts(seg['start'])} --> {_seconds_to_srt_ts(seg['end'])}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_seconds_to_vtt_ts(seg['start'])} --> {_seconds_to_vtt_ts(seg['end'])}")
        lines.append(seg.get("text", "").strip())
        lines.append("")
    return "\n".join(lines)


def _flatten_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        for w in seg.get("words", []) or []:
            entry: dict[str, Any] = {
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
            }
            if "speaker" in w:
                entry["speaker"] = w["speaker"]
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def _transcribe(
    audio_path: str,
    model_name: str,
    language: str | None,
    temperature: float,
    prompt: str | None,
    want_words: bool,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    audio = whisperx.load_audio(audio_path)

    model = _get_whisper(model_name)
    transcribe_kwargs: dict[str, Any] = {"batch_size": DEFAULT_BATCH_SIZE}
    if language:
        transcribe_kwargs["language"] = language
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt
    if temperature:
        transcribe_kwargs["temperature"] = temperature

    result = model.transcribe(audio, **transcribe_kwargs)
    detected_language = result.get("language", language or "")

    # Alignment (word-level timestamps). Required if the caller asked for word
    # granularity OR diarization (word-level speaker assignment needs word
    # timing).
    if want_words or diarize:
        try:
            align_model, align_metadata = _get_align(detected_language)
            result = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
        except Exception as exc:  # noqa: BLE001
            # Language may lack a wav2vec2 aligner. Degrade gracefully:
            # keep segment-level output, skip word timings + diarization.
            print(f"WARN: alignment failed for language={detected_language}: {exc}")
            want_words = False
            diarize = False
            result = {"segments": result["segments"]}

    speakers: list[str] = []
    if diarize:
        if _DIARIZE_PIPELINE is None:
            raise HTTPException(
                status_code=400,
                detail="diarization requested but pyannote pipeline is not available",
            )
        diar_kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            diar_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diar_kwargs["max_speakers"] = max_speakers
        diarize_segments = _DIARIZE_PIPELINE(audio, **diar_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        # Collect unique speaker labels in first-seen order.
        seen: set[str] = set()
        for seg in result.get("segments", []):
            spk = seg.get("speaker")
            if spk and spk not in seen:
                seen.add(spk)
                speakers.append(spk)

    segments: list[dict[str, Any]] = result.get("segments", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    return {
        "task": "transcribe",
        "language": detected_language,
        "duration": float(len(audio)) / 16000.0 if hasattr(audio, "__len__") else 0.0,
        "text": text,
        "segments": segments,
        "words": _flatten_words(segments) if want_words else None,
        "speakers": speakers if diarize else None,
    }


def _format_response(
    result: dict[str, Any],
    response_format: str,
    want_words: bool,
    diarize: bool,
):
    segments = result["segments"]
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    if response_format == "json":
        return JSONResponse({"text": result["text"]})

    # verbose_json — full payload, trimmed to what the caller asked for.
    payload: dict[str, Any] = {
        "task": result["task"],
        "language": result["language"],
        "duration": result["duration"],
        "text": result["text"],
        "segments": segments,
    }
    if want_words and result.get("words") is not None:
        payload["words"] = result["words"]
    if diarize and result.get("speakers"):
        payload["speakers"] = result["speakers"]
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Report the single served model id the client should send.

    The Whisper model is pinned at launch, so this reports exactly the id the
    endpoint accepts in the request `model` field: the SERVED_MODEL_NAME alias
    if configured, else the pinned DEFAULT_MODEL.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME or DEFAULT_MODEL,
                "object": "model",
                "owned_by": "openai-whisper",
            }
        ],
    }


async def _handle_transcription(
    file: UploadFile,
    model: str | None,
    language: str | None,
    response_format: str,
    temperature: float,
    prompt: str | None,
    timestamp_granularities: list[str] | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
):
    # Resolve the client-supplied `model` (or its absence) to the pinned model
    # before any work runs; rejects unknown ids unless override is enabled.
    model = _resolve_model(model)

    valid_formats = {"json", "text", "srt", "vtt", "verbose_json"}
    if response_format not in valid_formats:
        raise HTTPException(400, f"response_format must be one of {sorted(valid_formats)}")

    granularities = timestamp_granularities or ["segment"]
    want_words = "word" in granularities

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")

    # WhisperX + faster-whisper read from a path (ffmpeg is invoked as a
    # subprocess). Spool the upload to a NamedTemporaryFile so the ffmpeg
    # child can read it.
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = _transcribe(
            audio_path=tmp.name,
            model_name=model,
            language=language,
            temperature=temperature,
            prompt=prompt,
            want_words=want_words,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    return _format_response(result, response_format, want_words, diarize)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: str | None = Form(None),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    return await _handle_transcription(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        temperature=temperature,
        prompt=prompt,
        timestamp_granularities=timestamp_granularities,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@app.post("/invocations")
async def invocations(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: str | None = Form(None),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    return await _handle_transcription(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        temperature=temperature,
        prompt=prompt,
        timestamp_granularities=timestamp_granularities,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
