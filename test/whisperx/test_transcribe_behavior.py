"""Unit tests for the WhisperX server's transcription behavior.

Like ``test_server_options.py`` these run on CPU with no container and no GPU:
torch / whisperx / fastapi are stubbed in ``sys.modules`` before ``server.py``
is imported (``anyio`` and ``functools`` are the real thing — ``anyio`` ships
with fastapi/starlette). We then exercise two behaviors that are pure control
flow and need no model:

  * FIX B — the alignment fallback only degrades on the exceptions whisperX
    raises for a genuinely unsupported language (ValueError / NotImplementedError
    / KeyError). Any other error (e.g. a CUDA-OOM RuntimeError) propagates as a
    real failure instead of a silent HTTP 200, and an explicit ``diarize=true``
    that degrades becomes a 422 rather than a speaker-less 200.
  * FIX A — the blocking ``_transcribe`` call is offloaded off the event loop
    via ``anyio.to_thread.run_sync`` so long transcriptions don't starve /ping.
"""

import asyncio
import importlib.util
import inspect
import sys
import threading
import time
import types
from pathlib import Path

import pytest

# server.py lives in the image-build tree, a sibling of test/; load it by path.
SERVER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "docker" / "whisperx" / "server.py"


def _install_stubs() -> None:
    """Register lightweight fakes for the heavy imports server.py performs.

    Same shape as test_server_options.py, but the fastapi fakes carry real
    behavior we assert on: HTTPException stores ``status_code``/``detail`` (so
    the 422 path is checkable) and the response classes accept their payload.
    """

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    def _form_marker(default=None, **kwargs):
        return default

    class _HTTPException(Exception):
        def __init__(self, status_code=None, detail=None, **kwargs):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _JSONResponse:
        def __init__(self, content=None, **kwargs):
            self.content = content

    class _PlainTextResponse:
        def __init__(self, content=None, **kwargs):
            self.content = content

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeApp
    fastapi.File = _form_marker
    fastapi.Form = _form_marker
    fastapi.UploadFile = type("UploadFile", (), {})
    fastapi.HTTPException = _HTTPException

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.JSONResponse = _JSONResponse
    fastapi_responses.PlainTextResponse = _PlainTextResponse

    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = lambda *a, **k: None
    whisperx.load_audio = lambda *a, **k: None
    whisperx.load_align_model = lambda *a, **k: (None, {})
    whisperx.align = lambda *a, **k: {}
    whisperx.assign_word_speakers = lambda *a, **k: {}

    whisperx_diarize = types.ModuleType("whisperx.diarize")
    whisperx_diarize.DiarizationPipeline = type("DiarizationPipeline", (), {})
    whisperx.diarize = whisperx_diarize

    # server.py imports WriteSRT/WriteVTT from whisperx.utils at module scope;
    # stub the submodule so the import resolves (these tests never render subtitles).
    whisperx_utils = types.ModuleType("whisperx.utils")
    whisperx_utils.WriteSRT = type("WriteSRT", (), {})
    whisperx_utils.WriteVTT = type("WriteVTT", (), {})
    whisperx.utils = whisperx_utils

    sys.modules.update(
        {
            "torch": torch,
            "fastapi": fastapi,
            "fastapi.responses": fastapi_responses,
            "whisperx": whisperx,
            "whisperx.diarize": whisperx_diarize,
            "whisperx.utils": whisperx_utils,
        }
    )


def _load_server():
    """Import server.py fresh (stubs installed) so each test gets empty LRUs."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location("whisperx_server_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    """Stand-in Whisper model: transcribe() yields one Spanish segment."""

    def transcribe(self, audio, **kwargs):
        return {"segments": [{"text": "hola", "start": 0.0, "end": 1.0}], "language": "es"}


def _server_with_align_error(exc: Exception):
    """Load server with load_audio/_get_whisper stubbed and whisperx.align raising ``exc``."""
    server = _load_server()
    server.whisperx.load_audio = lambda path: [0.0] * 16000
    server._get_whisper = lambda name: _FakeModel()

    def _raise(*args, **kwargs):
        raise exc

    server.whisperx.align = _raise
    return server


def _transcribe(server, *, want_words: bool, diarize: bool):
    return server._transcribe(
        audio_path="/tmp/does-not-exist.wav",
        model_name="large-v2",
        language=None,
        want_words=want_words,
        diarize=diarize,
        min_speakers=None,
        max_speakers=None,
    )


# ---------------------------------------------------------------------------
# FIX B — narrowed alignment except
# ---------------------------------------------------------------------------
def test_alignment_runtime_error_propagates():
    """A non-degradation error (CUDA OOM) must bubble up, not become a 200."""
    server = _server_with_align_error(RuntimeError("CUDA OOM"))
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        _transcribe(server, want_words=True, diarize=False)


def test_unsupported_language_degrades_for_want_words():
    """A genuine no-aligner ValueError degrades a want_words request to segments."""
    server = _server_with_align_error(ValueError("No default align-model for language: xx"))
    result = _transcribe(server, want_words=True, diarize=False)
    assert isinstance(result, dict)
    assert result["segments"] == [{"text": "hola", "start": 0.0, "end": 1.0}]
    assert result["words"] is None  # degraded: no word timings, but no exception
    assert result["speakers"] is None
    assert result["language"] == "es"


def test_diarize_requested_but_alignment_unavailable_raises_422():
    """An explicit diarize=true that can't be aligned is a 422, not a silent 200."""
    server = _server_with_align_error(ValueError("No default align-model for language: xx"))
    with pytest.raises(server.HTTPException) as exc_info:
        _transcribe(server, want_words=False, diarize=True)
    assert exc_info.value.status_code == 422
    assert "diarization was requested" in exc_info.value.detail


def test_notimplementederror_also_degrades():
    """NotImplementedError (unsupported align model type) is a degrade case too."""
    server = _server_with_align_error(NotImplementedError("Align model of type ... not supported"))
    result = _transcribe(server, want_words=True, diarize=False)
    assert result["words"] is None
    assert result["segments"]


# ---------------------------------------------------------------------------
# FIX A — offload blocking work off the event loop
# ---------------------------------------------------------------------------
class _FakeUpload:
    filename = "audio.wav"

    async def read(self):
        return b"RIFFfake-audio-bytes"


def test_handle_transcription_is_coroutine_and_anyio_wired():
    """Smoke check that the offload path is present: async handler + anyio import."""
    server = _load_server()
    assert inspect.iscoroutinefunction(server._handle_transcription)
    assert hasattr(server, "anyio")
    assert "anyio" in sys.modules


def test_transcribe_runs_in_worker_thread():
    """The blocking _transcribe must run on a different thread than the loop."""
    server = _load_server()
    main_thread_id = threading.get_ident()
    recorded: dict = {}

    def _recorder(**kwargs):
        recorded["thread_id"] = threading.get_ident()
        recorded["kwargs"] = kwargs
        return {"text": "hi", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

    server._transcribe = _recorder

    asyncio.run(
        server._handle_transcription(
            file=_FakeUpload(),
            model=None,
            language=None,
            response_format="json",
            timestamp_granularities=None,
            diarize=False,
            min_speakers=None,
            max_speakers=None,
        )
    )

    assert recorded["thread_id"] != main_thread_id
    assert recorded["kwargs"]["audio_path"]  # tempfile path was forwarded through


# ---------------------------------------------------------------------------
# FIX C — cache getters dedupe concurrent loads under the worker-thread pool
# ---------------------------------------------------------------------------
def _run_concurrently(target, n=8):
    """Fire ``n`` threads at ``target`` at once; return once all have joined."""
    barrier = threading.Barrier(n)

    def _worker():
        barrier.wait()  # release all threads together to widen the race window
        target()

    threads = [threading.Thread(target=_worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_get_whisper_dedupes_concurrent_loads():
    """Concurrent misses for one key must trigger exactly one load (the lock).

    Without the per-cache lock, N worker threads all fail the membership check
    before any store completes and each calls load_model — N redundant ~30 s
    loads. The lock serializes check-load-store so only the first loads.
    """
    server = _load_server()
    calls = {"count": 0}
    lock = threading.Lock()
    sentinel = object()

    def _fake_load_model(*args, **kwargs):
        with lock:
            calls["count"] += 1
        time.sleep(0.05)  # widen the check-then-act window
        return sentinel

    server.whisperx.load_model = _fake_load_model

    results = []
    results_lock = threading.Lock()

    def _call():
        model = server._get_whisper("same-model")
        with results_lock:
            results.append(model)

    _run_concurrently(_call, n=8)

    assert calls["count"] == 1  # lock deduped 8 concurrent misses to one load
    assert len(results) == 8
    assert all(m is sentinel for m in results)


def test_get_align_dedupes_concurrent_loads():
    """Same dedupe guarantee for the align cache, which is keyed by language."""
    server = _load_server()
    calls = {"count": 0}
    lock = threading.Lock()
    sentinel = object()

    def _fake_load_align_model(*args, **kwargs):
        with lock:
            calls["count"] += 1
        time.sleep(0.05)  # widen the check-then-act window
        return (sentinel, {})

    server.whisperx.load_align_model = _fake_load_align_model

    results = []
    results_lock = threading.Lock()

    def _call():
        model, metadata = server._get_align("es")  # one language across threads
        with results_lock:
            results.append(model)

    _run_concurrently(_call, n=8)

    assert calls["count"] == 1  # lock deduped 8 concurrent misses to one load
    assert len(results) == 8
    assert all(m is sentinel for m in results)
