"""WhisperX FastAPI server.

Design ref: workspace/whisperx-docker/DESIGN.md §5.

Four routes, all served in one process:
    POST /v1/audio/transcriptions   — primary, OpenAI-compatible
    POST /invocations               — alias for SageMaker
    GET  /ping                      — readiness health check
    GET  /v1/models                 — advertises the single served model id

Extension fields on top of OpenAI's schema: `diarize`, `min_speakers`,
`max_speakers`. When `diarize=false` (default) output is byte-identical to
OpenAI's `verbose_json`.

Subtitle line-formatting fields — `max_line_width`, `max_line_count`,
`highlight_words` — are WhisperX-CLI-named (not OpenAI) and apply ONLY to the
`srt` and `vtt` response formats; for json/text/verbose_json they are accepted
but ignored (matching WhisperX, whose non-subtitle writers ignore them). Setting
any of them routes srt/vtt through WhisperX's own SubtitlesWriter so output is
byte-identical to the WhisperX CLI, and forces word-level alignment internally
(the writer only wraps/highlights when segments carry word timings).

All WhisperX model, decoding, and VAD parameters (beam size, temperature,
initial prompt, VAD thresholds, task, ...) are fixed at container launch via
WHISPERX_* env vars — see the "Launch-time WhisperX configuration" block below.
The HTTP API is deliberately lean: `temperature` and `prompt` are launch-only
knobs, not per-request fields.

The Whisper model is one-per-container, chosen at launch via
WHISPERX_DEFAULT_MODEL; the request `model` field is accepted but ignored (an
OpenAI-compat no-op), so a single model stays resident rather than being loaded
per request.
"""

from __future__ import annotations

import functools
import io
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import torch
import whisperx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import WriteSRT, WriteVTT


# ---------------------------------------------------------------------------
# Launch-time configuration helpers
# ---------------------------------------------------------------------------
# All WhisperX decoding/VAD/model knobs are fixed at container launch via env
# vars and read once (below) into immutable module globals. Grouped by where
# whisperx.load_model consumes them:
#
#   asr_options (decoding) — WHISPERX_BEAM_SIZE, WHISPERX_BEST_OF,
#     WHISPERX_PATIENCE, WHISPERX_LENGTH_PENALTY, WHISPERX_TEMPERATURE,
#     WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK,
#     WHISPERX_COMPRESSION_RATIO_THRESHOLD, WHISPERX_LOGPROB_THRESHOLD,
#     WHISPERX_NO_SPEECH_THRESHOLD, WHISPERX_CONDITION_ON_PREVIOUS_TEXT,
#     WHISPERX_INITIAL_PROMPT, WHISPERX_HOTWORDS, WHISPERX_SUPPRESS_TOKENS,
#     WHISPERX_SUPPRESS_NUMERALS
#   vad_options — WHISPERX_CHUNK_SIZE, WHISPERX_VAD_ONSET, WHISPERX_VAD_OFFSET
#   pipeline/model — WHISPERX_VAD_METHOD, WHISPERX_TASK, WHISPERX_ALIGN_MODEL,
#     WHISPERX_DEFAULT_MODEL, WHISPERX_COMPUTE_TYPE, WHISPERX_BATCH_SIZE,
#     WHISPERX_SERVED_MODEL_NAME, ...
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _build_asr_options() -> dict[str, Any]:
    """Assemble faster-whisper decoding options from WHISPERX_* env.

    Mirrors the whisperX CLI defaults; passed to load_model(asr_options=...) at
    launch. Pure (env -> dict) so it is unit-testable without a GPU/container.
    """
    base = float(os.environ.get("WHISPERX_TEMPERATURE") or 0.0)
    inc_raw = os.environ.get("WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK", "0.2")
    inc = float(inc_raw) if inc_raw.strip() else 0.0
    if inc > 0:
        # Match whisper's np.arange(base, 1.0 + 1e-6, inc) fallback schedule.
        steps = int((1.0 + 1e-6 - base) / inc) + 1
        temperatures = tuple(round(base + i * inc, 4) for i in range(max(steps, 1)))
    else:
        temperatures = (base,)
    suppress = os.environ.get("WHISPERX_SUPPRESS_TOKENS", "-1")
    return {
        "beam_size": int(os.environ.get("WHISPERX_BEAM_SIZE", "5")),
        "best_of": int(os.environ.get("WHISPERX_BEST_OF", "5")),
        "patience": float(os.environ.get("WHISPERX_PATIENCE", "1.0")),
        "length_penalty": float(os.environ.get("WHISPERX_LENGTH_PENALTY", "1.0")),
        "temperatures": temperatures,
        "compression_ratio_threshold": float(
            os.environ.get("WHISPERX_COMPRESSION_RATIO_THRESHOLD", "2.4")
        ),
        "log_prob_threshold": float(os.environ.get("WHISPERX_LOGPROB_THRESHOLD", "-1.0")),
        "no_speech_threshold": float(os.environ.get("WHISPERX_NO_SPEECH_THRESHOLD", "0.6")),
        "condition_on_previous_text": _env_bool("WHISPERX_CONDITION_ON_PREVIOUS_TEXT"),
        "initial_prompt": os.environ.get("WHISPERX_INITIAL_PROMPT") or None,
        "hotwords": os.environ.get("WHISPERX_HOTWORDS") or None,
        "suppress_tokens": [int(x) for x in suppress.split(",") if x.strip()],
        "suppress_numerals": _env_bool("WHISPERX_SUPPRESS_NUMERALS"),
    }


def _build_vad_options() -> dict[str, Any]:
    """Assemble VAD options from WHISPERX_* env; passed to load_model(vad_options=...)."""
    return {
        "chunk_size": int(os.environ.get("WHISPERX_CHUNK_SIZE", "30")),
        "vad_onset": float(os.environ.get("WHISPERX_VAD_ONSET", "0.5")),
        "vad_offset": float(os.environ.get("WHISPERX_VAD_OFFSET", "0.363")),
    }


# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
# One Whisper model per container, pinned at launch (DEFAULT_MODEL) and warmed
# in the lifespan hook. The request `model` field is ignored (OpenAI-compat
# no-op), so a single model stays resident rather than a per-name LRU.
_WHISPER_MODEL: Any = None

# wav2vec2 align models — keyed by language code. Orthogonal to the Whisper
# model (a request's detected language selects the aligner), so this stays an
# LRU even though only one Whisper model is served.
_ALIGN_LRU: "OrderedDict[str, tuple[Any, dict[str, Any]]]" = OrderedDict()
_ALIGN_LRU_MAX = int(os.environ.get("WHISPERX_ALIGN_LRU_SIZE", "3"))

# _transcribe runs in an anyio worker thread. _WHISPER_LOCK serializes the
# one-time model load (defense-in-depth if MAX_CONCURRENT_REQUESTS>1 ever races
# concurrent cold misses, so only the first thread loads); _ALIGN_LOCK does the
# same for the align LRU. Inference itself stays outside these locks (it runs in
# _transcribe after the getters return).
_WHISPER_LOCK = threading.Lock()
_ALIGN_LOCK = threading.Lock()

# Fixed models loaded at startup.
_DIARIZE_PIPELINE: Any = None

# Process readiness. Starts True; the inference path flips it False on a fatal
# CUDA/device fault (see _is_fatal_cuda_error) so /ping can report 503.
_HEALTHY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
# The Whisper model pinned at container launch and warmed in the lifespan hook.
DEFAULT_MODEL = os.environ.get("WHISPERX_DEFAULT_MODEL", "large-v2")
# Optional client-facing alias for the pinned model (e.g. "whisper-1" for an
# OpenAI-SDK drop-in). Used ONLY by /v1/models to advertise the served id; the
# request `model` field is ignored, so no request-time resolution happens.
SERVED_MODEL_NAME = os.environ.get("WHISPERX_SERVED_MODEL_NAME")
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
DIARIZE_MODEL_PATH = os.environ.get(
    "WHISPERX_DIARIZE_MODEL_PATH",
    "/opt/models/pyannote/speaker-diarization-community-1",
)
# VAD backend and transcription task are load_model args (not per-request). An
# optional align model override lets operators pin a specific wav2vec2 model.
VAD_METHOD = os.environ.get("WHISPERX_VAD_METHOD", "pyannote")
TASK = os.environ.get("WHISPERX_TASK", "transcribe")
ALIGN_MODEL = os.environ.get("WHISPERX_ALIGN_MODEL") or None
# Decoding + VAD options are global launch config: read env once here so the
# model loaded via _get_whisper uses one immutable configuration.
ASR_OPTIONS = _build_asr_options()
VAD_OPTIONS = _build_vad_options()

# ---------------------------------------------------------------------------
# Request admission + inference concurrency
# ---------------------------------------------------------------------------
# GPU inference is serialized to MAX_CONCURRENT_REQUESTS (default 1): WhisperX's
# FasterWhisperPipeline mutates self.options/self.tokenizer mid-call and the
# shared diarization pipeline is not documented thread-safe, so running two
# transcriptions on one instance concurrently is unsafe. _INFERENCE_LIMITER caps
# in-flight inference; MAX_QUEUE bounds how many requests may wait behind it
# before new ones are shed with 503; MAX_UPLOAD_BYTES bounds per-request
# memory/temp-file retention.
MAX_CONCURRENT_REQUESTS = int(os.environ.get("WHISPERX_MAX_CONCURRENT_REQUESTS", "1"))
_INFERENCE_LIMITER = anyio.CapacityLimiter(MAX_CONCURRENT_REQUESTS)
MAX_QUEUE = int(os.environ.get("WHISPERX_MAX_QUEUE", "2"))
MAX_UPLOAD_BYTES = int(os.environ.get("WHISPERX_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))


def _lru_touch(cache: "OrderedDict[str, Any]", key: str, max_size: int) -> None:
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _get_whisper(model_name: str) -> Any:
    # One model per container: model_name is always DEFAULT_MODEL, so a single
    # resident model is cached. _WHISPER_LOCK holds across the check-load-store
    # so that if MAX_CONCURRENT_REQUESTS>1 ever races two cold misses, only the
    # first loads (dedupes the ~30 s load); a warm hit takes the lock briefly.
    # Decoding/VAD/task config is global launch config (ASR_OPTIONS/VAD_OPTIONS).
    global _WHISPER_MODEL
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = whisperx.load_model(
                model_name,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                asr_options=ASR_OPTIONS,
                vad_method=VAD_METHOD,
                vad_options=VAD_OPTIONS,
                task=TASK,
            )
        return _WHISPER_MODEL


def _get_align(language: str) -> tuple[Any, dict[str, Any]]:
    # Same single-critical-section rationale as _get_whisper: serialize the
    # keyed-by-language check-load-store-touch so concurrent worker threads
    # dedupe the load and never race move_to_end against an eviction.
    with _ALIGN_LOCK:
        if language not in _ALIGN_LRU:
            _ALIGN_LRU[language] = whisperx.load_align_model(
                language_code=language, device=DEVICE, model_name=ALIGN_MODEL
            )
        _lru_touch(_ALIGN_LRU, language, _ALIGN_LRU_MAX)
        return _ALIGN_LRU[language]


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load fixed models before uvicorn binds so /ping stays shallow but honest.

    Design §6 decision 8: `/ping` reachable ⇒ the diarization pipeline and the
    default Whisper model are resident. VAD uses whisperx's default (pyannote),
    whose segmentation model ships inside the whisperx wheel.
    """
    global _DIARIZE_PIPELINE, _HEALTHY
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

    # Warmup completed without a fatal load error; confirm readiness. (If
    # _get_whisper raised above, we never reach here and uvicorn won't bind.)
    _HEALTHY = True
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


def _render_subtitle(
    result: dict[str, Any],
    fmt: str,
    max_line_width: int | None,
    max_line_count: int | None,
    highlight_words: bool,
) -> str:
    """Render srt/vtt via WhisperX's own SubtitlesWriter (byte-identical to the CLI).

    Reuses whisperx.utils.WriteSRT / WriteVTT so line wrapping and word
    highlighting exactly match the WhisperX CLI. output_dir is only consumed by
    the writer's __call__ (file-path mode), never by write_result, so "" is safe.
    All three option keys are mandatory: SubtitlesWriter.iterate_result reads
    each by key and KeyErrors otherwise. `result` must carry "language" (checked
    against LANGUAGES_WITHOUT_SPACES) and "segments"; pass the full _transcribe
    result, not just segments. When segments lack "words" (alignment didn't run
    or degraded for an unsupported language) the writer falls back to
    segment-level cues automatically, so the knobs simply no-op.
    """
    writer = (WriteSRT if fmt == "srt" else WriteVTT)(output_dir="")
    options = {
        "max_line_width": max_line_width,
        "max_line_count": max_line_count,
        "highlight_words": highlight_words,
    }
    buf = io.StringIO()
    writer.write_result(result, file=buf, options=options)
    return buf.getvalue()


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
# Fatal CUDA/device error substrings — a RuntimeError whose message contains any
# of these means the process's GPU context is corrupted and every subsequent
# request will fail, so readiness flips. Transient faults (CUDA OOM) are
# deliberately excluded: they don't poison the context.
_FATAL_CUDA_MARKERS = (
    "cuda error",
    "device-side assert",
    "illegal memory access",
    "misaligned address",
    "unspecified launch failure",
)


def _is_fatal_cuda_error(exc: Exception) -> bool:
    """True for unrecoverable CUDA/device faults that corrupt the GPU context.

    Only RuntimeErrors carrying a known-fatal marker qualify. A plain CUDA OOM
    (torch.cuda.OutOfMemoryError / "out of memory") is transient — the context
    survives and later requests can succeed — so it is NOT treated as fatal.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_CUDA_MARKERS)


def _transcribe(
    audio_path: str,
    language: str | None,
    want_words: bool,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    audio = whisperx.load_audio(audio_path)

    model = _get_whisper(DEFAULT_MODEL)
    # Decoding params (temperature/prompt/beam/...) are baked into the model at
    # load time via ASR_OPTIONS. FasterWhisperPipeline.transcribe only accepts
    # pipeline-level args; passing decoding kwargs here raises TypeError. `task`
    # is passed per-call because load_model discards its task= when no language
    # is pinned at load time, so it must reach transcribe() to take effect.
    transcribe_kwargs: dict[str, Any] = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "chunk_size": VAD_OPTIONS["chunk_size"],
        "task": TASK,
    }
    if language:
        transcribe_kwargs["language"] = language

    try:
        result = model.transcribe(audio, **transcribe_kwargs)
        detected_language = result.get("language", language or "")

        # Alignment (word-level timestamps). Required if the caller asked for
        # word granularity OR diarization (word-level speaker assignment needs
        # word timing).
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
            except (ValueError, NotImplementedError, KeyError) as exc:
                # These are the only exceptions whisperX raises when a language
                # genuinely has no wav2vec2 aligner (no default align-model /
                # unsupported model type / missing metadata key). Only these
                # degrade gracefully; any other error (CUDA OOM RuntimeError,
                # network failure fetching the aligner, ...) propagates so it
                # surfaces as a real 500 instead of a silently-degraded 200.
                print(f"WARN: alignment failed for language={detected_language}: {exc}")
                degraded_from_diarize = diarize
                want_words = False
                diarize = False
                result = {"segments": result["segments"]}
                # Diarization needs word timing, so degradation makes the
                # requested speakers impossible to produce. Rather than return
                # 200 with no speakers, tell the caller explicitly. A best-effort
                # want_words request keeps the silent (logged) fallback above.
                if degraded_from_diarize:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"diarization was requested but alignment is unavailable for "
                            f"language '{detected_language}' (no wav2vec2 aligner); retry "
                            f"without diarize or with a supported language"
                        ),
                    ) from exc

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
    except Exception as exc:
        # A fatal CUDA/device fault corrupts this process's GPU context, so every
        # later request on this instance will fail too. Flip readiness (=> /ping
        # 503) so orchestration can drain/replace the host, then re-raise so the
        # current request still errors. Transient failures — CUDA OOM, the
        # unsupported-language degrade above, and the 422/400 HTTPExceptions — are
        # NOT fatal and simply propagate without flipping health.
        global _HEALTHY
        if _is_fatal_cuda_error(exc):
            _HEALTHY = False
        raise

    segments: list[dict[str, Any]] = result.get("segments", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    return {
        "task": TASK,
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
    max_line_width: int | None = None,
    max_line_count: int | None = None,
    highlight_words: bool = False,
):
    segments = result["segments"]
    # A subtitle knob being set routes srt/vtt through WhisperX's SubtitlesWriter
    # (byte-identical to the CLI). With no knob set we keep the legacy
    # _to_srt/_to_vtt so default srt/vtt output is unchanged. The knobs are
    # subtitle-only: json/text/verbose_json ignore them (mirroring WhisperX).
    subtitle_knob = max_line_width is not None or max_line_count is not None or highlight_words
    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format == "srt":
        if subtitle_knob:
            body = _render_subtitle(result, "srt", max_line_width, max_line_count, highlight_words)
        else:
            body = _to_srt(segments)
        return PlainTextResponse(body, media_type="application/x-subrip")
    if response_format == "vtt":
        if subtitle_knob:
            body = _render_subtitle(result, "vtt", max_line_width, max_line_count, highlight_words)
        else:
            body = _to_vtt(segments)
        return PlainTextResponse(body, media_type="text/vtt")
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
def ping():
    # Readiness reflects fatal CUDA/model state observed by the inference path.
    # It is PASSIVE: a GPU that dies while the server is idle won't be caught
    # here (no request has touched the device) — that's host-monitoring's job.
    # Once the inference path hits a fatal CUDA fault, _HEALTHY flips and every
    # /ping returns 503 so orchestration can drain/replace the instance.
    if _HEALTHY:
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Advertise the single served model id for OpenAI-client compatibility.

    The Whisper model is one-per-container and the request `model` field is
    ignored, so this simply reports the id a client should display/send: the
    SERVED_MODEL_NAME alias if configured, else the pinned DEFAULT_MODEL.
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
    language: str | None,
    response_format: str,
    timestamp_granularities: list[str] | None,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
    highlight_words: bool = False,
):
    valid_formats = {"json", "text", "srt", "vtt", "verbose_json"}
    if response_format not in valid_formats:
        raise HTTPException(400, f"response_format must be one of {sorted(valid_formats)}")

    # Validate the extension params at the boundary so a bad request fails fast
    # with a 422 instead of surfacing deep in the diarization/subtitle path.
    if min_speakers is not None and min_speakers < 1:
        raise HTTPException(422, "min_speakers must be >= 1")
    if max_speakers is not None and max_speakers < 1:
        raise HTTPException(422, "max_speakers must be >= 1")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise HTTPException(422, "min_speakers must be <= max_speakers")
    if max_line_width is not None and max_line_width < 0:
        raise HTTPException(422, "max_line_width must be >= 0")
    if max_line_count is not None and max_line_count < 0:
        raise HTTPException(422, "max_line_count must be >= 0")

    granularities = timestamp_granularities or ["segment"]
    want_words = "word" in granularities

    # Subtitle line-formatting applies only to srt/vtt. WhisperX's
    # SubtitlesWriter only wraps/highlights when segments carry word timings, so
    # a subtitle knob forces word-level alignment internally (independent of the
    # user-facing timestamp_granularities). Leave want_words as derived above and
    # only OR in this internal flag for the _transcribe call.
    subtitle_formatting = response_format in {"srt", "vtt"} and (
        max_line_width is not None or max_line_count is not None or highlight_words
    )

    # Admission control: when the inference queue is already full, shed load with
    # 503 before the full-size `await file.read()` below, so an overload doesn't
    # materialize more large byte buffers or grow the worker queue. This is a soft
    # bound, not a hard cap: Starlette's File(...) has already spooled the raw
    # multipart body by the time this runs, and tasks_waiting only counts requests
    # that have reached the limiter, so a tight concurrent burst can admit a few
    # past this check. Excess admitted requests then wait on the event loop.
    if _INFERENCE_LIMITER.statistics().tasks_waiting >= MAX_QUEUE:
        raise HTTPException(503, "server busy: inference queue full")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio upload")
    # Bound per-request memory + temp-file retention. Streaming the upload to
    # disk with an incremental size check is a future improvement; this rejects
    # outsized uploads after a single read.
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")

    # WhisperX + faster-whisper read from a path (ffmpeg is invoked as a
    # subprocess). Spool the upload to a NamedTemporaryFile so the ffmpeg
    # child can read it.
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        # _transcribe is fully synchronous and long-running (GPU inference plus
        # an ffmpeg subprocess). Running it inline would block uvicorn's single
        # event loop for the whole transcription, starving every other
        # coroutine — including GET /ping — and risking SageMaker health-check
        # failures. Offload it to a worker thread so the loop stays responsive.
        # The limiter serializes the whole _transcribe (transcription + align +
        # diarization): WhisperX's FasterWhisperPipeline mutates
        # self.options/self.tokenizer mid-call and the shared diarization
        # pipeline is not documented thread-safe, so concurrent inference on one
        # instance is unsafe. MAX_CONCURRENT_REQUESTS (default 1) caps it; excess
        # requests wait on the event loop (not in the worker pool) so /ping stays
        # responsive.
        result = await anyio.to_thread.run_sync(
            functools.partial(
                _transcribe,
                audio_path=tmp.name,
                language=language,
                want_words=want_words or subtitle_formatting,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            ),
            limiter=_INFERENCE_LIMITER,
        )

    return _format_response(
        result,
        response_format,
        want_words,
        diarize,
        max_line_width,
        max_line_count,
        highlight_words,
    )


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    max_line_width: int | None = Form(None),
    max_line_count: int | None = Form(None),
    highlight_words: bool = Form(False),
):
    return await _handle_transcription(
        file=file,
        language=language,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
        highlight_words=highlight_words,
    )


@app.post("/invocations")
async def invocations(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    max_line_width: int | None = Form(None),
    max_line_count: int | None = Form(None),
    highlight_words: bool = Form(False),
):
    return await _handle_transcription(
        file=file,
        language=language,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
        highlight_words=highlight_words,
    )
