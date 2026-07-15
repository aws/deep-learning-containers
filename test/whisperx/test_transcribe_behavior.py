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

Later review findings covered here (all GPU-free control flow):

  * FINDING #1 — GPU inference is serialized via ``_INFERENCE_LIMITER``
    (``WHISPERX_MAX_CONCURRENT_REQUESTS``, default 1), so two concurrent
    ``_handle_transcription`` calls never run ``_transcribe`` at the same time.
  * FINDING #3 — the returned dict's ``task`` reflects ``WHISPERX_TASK``.
  * FINDING #4 — admission control: a full inference queue is shed with 503
    before the body is read, and an oversized upload is rejected with 413.
  * FINDING #5 — passive readiness: ``/ping`` reports 503 once the inference
    path observes a fatal CUDA fault; transient OOM leaves it healthy.
  * FINDING #6 — extension params are validated at the boundary (422).
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
            self.status_code = kwargs.get("status_code", 200)
            self.media_type = kwargs.get("media_type")

    class _PlainTextResponse:
        def __init__(self, content=None, **kwargs):
            self.content = content
            self.media_type = kwargs.get("media_type")

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
    """Concurrent cold misses must trigger exactly one model load (the lock).

    One model per container: _get_whisper caches a single resident model. Without
    _WHISPER_LOCK, N worker threads racing the first request would all see
    _WHISPER_MODEL is None before any store completes and each call load_model —
    N redundant ~30 s loads. The lock serializes check-load-store so only the
    first loads (defense-in-depth if MAX_CONCURRENT_REQUESTS>1).
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
        model = server._get_whisper("large-v2")
        with results_lock:
            results.append(model)

    _run_concurrently(_call, n=8)

    assert calls["count"] == 1  # lock deduped 8 concurrent misses to one load
    assert len(results) == 8
    assert all(m is sentinel for m in results)


def _handle(server, upload=None, **overrides):
    """Drive _handle_transcription with sane defaults; ``overrides`` win."""
    kwargs = {
        "file": upload or _FakeUpload(),
        "language": None,
        "response_format": "json",
        "timestamp_granularities": None,
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
    }
    kwargs.update(overrides)
    return asyncio.run(server._handle_transcription(**kwargs))


# ---------------------------------------------------------------------------
# FINDING #2 — one model per container: the `model` field left the request path
# ---------------------------------------------------------------------------
def test_model_field_removed_from_request_path():
    """`model` is gone from every request-path signature, and so is override."""
    server = _load_server()
    for fn in (server._handle_transcription, server.transcribe, server.invocations):
        assert "model" not in inspect.signature(fn).parameters
    assert "model_name" not in inspect.signature(server._transcribe).parameters
    # The override machinery is fully deleted.
    assert not hasattr(server, "_resolve_model")
    assert not hasattr(server, "ALLOW_MODEL_OVERRIDE")


# ---------------------------------------------------------------------------
# FINDING #3 — the returned dict's task reflects WHISPERX_TASK, not a literal
# ---------------------------------------------------------------------------
def test_transcribe_task_reflects_env(monkeypatch):
    """WHISPERX_TASK=translate => result['task'] == 'translate' (was hardcoded)."""
    monkeypatch.setenv("WHISPERX_TASK", "translate")
    server = _load_server()
    server.whisperx.load_audio = lambda path: [0.0] * 16000
    server._get_whisper = lambda name: _FakeModel()

    result = _transcribe(server, want_words=False, diarize=False)
    assert server.TASK == "translate"
    assert result["task"] == "translate"


# ---------------------------------------------------------------------------
# FINDING #1 — GPU inference is serialized to one at a time (default capacity 1)
# ---------------------------------------------------------------------------
def test_inference_serialized_to_capacity_one():
    """Two concurrent handlers must never run _transcribe at the same time.

    The stubbed _transcribe tracks a live-count with a lock and sleeps; if the
    CapacityLimiter (default 1) serializes correctly the max observed overlap is
    exactly 1. Without it the two worker threads would both be inside at once.
    """
    server = _load_server()
    live = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def _blocking(**kwargs):
        with lock:
            live["cur"] += 1
            live["max"] = max(live["max"], live["cur"])
        time.sleep(0.05)  # hold the slot so an overlap would be observable
        with lock:
            live["cur"] -= 1
        return {"text": "hi", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

    server._transcribe = _blocking

    async def _drive():
        await asyncio.gather(_run_one(), _run_one())

    async def _run_one():
        await server._handle_transcription(
            file=_FakeUpload(),
            language=None,
            response_format="json",
            timestamp_granularities=None,
            diarize=False,
            min_speakers=None,
            max_speakers=None,
        )

    asyncio.run(_drive())
    assert live["max"] == 1  # the limiter prevented any overlap


# ---------------------------------------------------------------------------
# FINDING #4 — admission control: shed a full queue (503) and cap upload (413)
# ---------------------------------------------------------------------------
def test_full_queue_returns_503_before_reading_body():
    """tasks_waiting >= MAX_QUEUE => 503 raised BEFORE the upload is read."""
    server = _load_server()

    class _FakeLimiter:
        def statistics(self):
            return types.SimpleNamespace(tasks_waiting=99)

    server._INFERENCE_LIMITER = _FakeLimiter()

    read_called = {"v": False}

    class _WatchedUpload:
        filename = "audio.wav"

        async def read(self):
            read_called["v"] = True
            return b"RIFFfake-audio-bytes"

    with pytest.raises(server.HTTPException) as exc_info:
        _handle(server, upload=_WatchedUpload())
    assert exc_info.value.status_code == 503
    assert read_called["v"] is False  # body never read under overload


def test_oversized_upload_returns_413(monkeypatch):
    """A body larger than WHISPERX_MAX_UPLOAD_BYTES is rejected 413."""
    monkeypatch.setenv("WHISPERX_MAX_UPLOAD_BYTES", "8")
    server = _load_server()
    assert server.MAX_UPLOAD_BYTES == 8

    class _BigUpload:
        filename = "audio.wav"

        async def read(self):
            return b"x" * 64  # exceeds the 8-byte cap

    with pytest.raises(server.HTTPException) as exc_info:
        _handle(server, upload=_BigUpload())
    assert exc_info.value.status_code == 413


# ---------------------------------------------------------------------------
# FINDING #5 — passive readiness health reflected by /ping
# ---------------------------------------------------------------------------
def test_ping_healthy_returns_200():
    server = _load_server()
    assert server._HEALTHY is True
    resp = server.ping()
    assert resp.status_code == 200
    assert resp.content == {"status": "ok"}


def test_ping_unhealthy_returns_503():
    server = _load_server()
    server._HEALTHY = False
    resp = server.ping()
    assert resp.status_code == 503
    assert resp.content == {"status": "unavailable"}


def test_is_fatal_cuda_error_classification():
    server = _load_server()
    assert server._is_fatal_cuda_error(RuntimeError("CUDA error: device-side assert triggered"))
    assert server._is_fatal_cuda_error(RuntimeError("out of memory")) is False
    assert server._is_fatal_cuda_error(ValueError("x")) is False


def test_fatal_cuda_error_flips_health_and_reraises():
    """A fatal CUDA fault in inference flips _HEALTHY False and still errors."""
    server = _load_server()
    server.whisperx.load_audio = lambda path: [0.0] * 16000

    class _BoomModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("CUDA error: illegal memory access was encountered")

    server._get_whisper = lambda name: _BoomModel()
    assert server._HEALTHY is True
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _transcribe(server, want_words=False, diarize=False)
    assert server._HEALTHY is False


def test_transient_oom_does_not_flip_health():
    """A plain CUDA OOM is transient: it propagates but leaves readiness intact."""
    server = _load_server()
    server.whisperx.load_audio = lambda path: [0.0] * 16000

    class _OomModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("CUDA out of memory")

    server._get_whisper = lambda name: _OomModel()
    with pytest.raises(RuntimeError, match="out of memory"):
        _transcribe(server, want_words=False, diarize=False)
    assert server._HEALTHY is True


# ---------------------------------------------------------------------------
# FINDING #6 — parameter validation at the boundary (422)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        {"min_speakers": 0},
        {"max_speakers": 0},
        {"min_speakers": 3, "max_speakers": 2},
        {"max_line_width": -1},
        {"max_line_count": -1},
    ],
)
def test_invalid_params_return_422(overrides):
    server = _load_server()

    def _must_not_run(**kwargs):
        raise AssertionError("_transcribe ran despite invalid params")

    server._transcribe = _must_not_run
    with pytest.raises(server.HTTPException) as exc_info:
        _handle(server, **overrides)
    assert exc_info.value.status_code == 422


def test_valid_params_reach_transcribe():
    """A valid speaker/line combo passes validation and reaches _transcribe."""
    server = _load_server()
    reached = {"v": False}

    def _recorder(**kwargs):
        reached["v"] = True
        return {"text": "hi", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

    server._transcribe = _recorder
    _handle(server, min_speakers=1, max_speakers=2, max_line_width=40, max_line_count=2)
    assert reached["v"] is True


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
