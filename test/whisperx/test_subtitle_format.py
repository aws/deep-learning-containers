"""Unit tests for the WhisperX server's subtitle line-formatting support.

Like ``test_transcribe_behavior.py`` these run on CPU with no container and no
GPU: torch / whisperx / fastapi are stubbed in ``sys.modules`` before
``server.py`` is imported. server.py now does ``from whisperx.utils import
WriteSRT, WriteVTT``, so the stub adds a ``whisperx.utils`` module whose
WriteSRT / WriteVTT are *recording* fakes: they capture the ``(result,
options)`` handed to ``write_result`` and emit a sentinel string.

The actual line-wrapping / word-highlighting algorithm is WhisperX's own
``SubtitlesWriter`` — not ours — so we do NOT reimplement or assert on it (that
would only test a fake). Instead we test the logic we wrote:

  * ``_render_subtitle`` builds the correct writer class for the format and
    passes an options dict carrying ALL THREE keys with the exact values.
  * ``_format_response`` routes srt/vtt through ``_render_subtitle`` only when a
    subtitle knob is set, and through the legacy ``_to_srt`` / ``_to_vtt``
    otherwise (so default srt/vtt output is unchanged), and json/text ignore the
    knobs entirely (no error).
  * ``_handle_transcription`` forces word-level alignment (want_words=True) when
    a subtitle knob is set on srt/vtt, and leaves want_words as derived
    otherwise (OR, never overwrite).

The real WhisperX writers emit outputs that satisfy the existing EC2 format
assertions — WriteVTT prints ``"WEBVTT\\n"`` first (whisperx/utils.py:377) and
WriteSRT prints ``-->`` (whisperx/utils.py:391), verified against whisperx
3.8.6 source. ``_format_response`` returns that output verbatim (proven here via
the sentinel passthrough), so the knob path keeps satisfying those assertions.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

# server.py lives in the image-build tree, a sibling of test/; load it by path.
SERVER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "docker" / "whisperx" / "server.py"

_SRT_SENTINEL = "<<WRITESRT-OUTPUT>>"
_VTT_SENTINEL = "<<WRITEVTT-OUTPUT>>"


def _make_recording_writer(name: str, sentinel: str):
    """Build a fake SubtitlesWriter that records calls and writes ``sentinel``.

    Records construction ``output_dir`` and every ``write_result(result,
    options)`` on class-level lists. Fresh classes are created per stub install,
    so the lists start empty for each server import.
    """

    class _RecordingWriter:
        writer_name = name
        init_output_dirs: list = []
        write_calls: list = []

        def __init__(self, output_dir=None, **kwargs):
            type(self).init_output_dirs.append(output_dir)

        def write_result(self, result, file, options):
            type(self).write_calls.append({"result": result, "options": options})
            file.write(sentinel)

    return _RecordingWriter


def _install_stubs() -> None:
    """Register lightweight fakes for the heavy imports server.py performs.

    Same shape as test_transcribe_behavior.py, plus a ``whisperx.utils`` module
    whose WriteSRT / WriteVTT are recording fakes. The response stubs also
    capture ``media_type`` so the srt/vtt content-type routing is checkable.
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

    whisperx_utils = types.ModuleType("whisperx.utils")
    whisperx_utils.WriteSRT = _make_recording_writer("WriteSRT", _SRT_SENTINEL)
    whisperx_utils.WriteVTT = _make_recording_writer("WriteVTT", _VTT_SENTINEL)
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


def _result() -> dict:
    """A minimal _transcribe-shaped result carrying language + segments + words."""
    return {
        "task": "transcribe",
        "language": "en",
        "duration": 1.0,
        "text": "hello world",
        "segments": [
            {
                "text": "hello world",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ],
            }
        ],
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ],
        "speakers": None,
    }


# ---------------------------------------------------------------------------
# _render_subtitle — writer selection + options assembly
# ---------------------------------------------------------------------------
def test_render_subtitle_srt_passes_all_three_options():
    """srt picks WriteSRT and forwards all three option keys with exact values."""
    server = _load_server()
    result = _result()
    out = server._render_subtitle(result, "srt", 42, 2, True)

    assert out == _SRT_SENTINEL  # verbatim writer output flows back
    assert server.WriteVTT.write_calls == []  # wrong writer never used
    assert len(server.WriteSRT.write_calls) == 1
    call = server.WriteSRT.write_calls[0]
    assert call["options"] == {
        "max_line_width": 42,
        "max_line_count": 2,
        "highlight_words": True,
    }
    assert call["result"] is result  # full result dict (has language + segments)
    assert server.WriteSRT.init_output_dirs == [""]  # output_dir unused by write_result


def test_render_subtitle_vtt_selects_vtt_writer_with_none_options():
    """vtt picks WriteVTT; None/None/False knobs still yield all three keys."""
    server = _load_server()
    out = server._render_subtitle(_result(), "vtt", None, None, False)

    assert out == _VTT_SENTINEL
    assert server.WriteSRT.write_calls == []
    assert len(server.WriteVTT.write_calls) == 1
    assert server.WriteVTT.write_calls[0]["options"] == {
        "max_line_width": None,
        "max_line_count": None,
        "highlight_words": False,
    }


# ---------------------------------------------------------------------------
# _format_response — routing between the WhisperX writer and legacy formatters
# ---------------------------------------------------------------------------
def test_format_response_srt_with_knob_uses_render_subtitle():
    """A knob on srt routes through _render_subtitle (WhisperX writer)."""
    server = _load_server()
    resp = server._format_response(
        _result(), "srt", want_words=True, diarize=False, max_line_width=42
    )
    assert len(server.WriteSRT.write_calls) == 1
    assert resp.content == _SRT_SENTINEL
    assert resp.media_type == "application/x-subrip"


def test_format_response_vtt_with_knob_uses_render_subtitle():
    """A knob on vtt routes through _render_subtitle with the vtt media type."""
    server = _load_server()
    resp = server._format_response(
        _result(), "vtt", want_words=True, diarize=False, highlight_words=True
    )
    assert len(server.WriteVTT.write_calls) == 1
    assert resp.content == _VTT_SENTINEL
    assert resp.media_type == "text/vtt"


def test_format_response_srt_without_knob_uses_legacy():
    """No knob on srt keeps the legacy _to_srt output (WhisperX writer untouched)."""
    server = _load_server()
    resp = server._format_response(_result(), "srt", want_words=False, diarize=False)
    assert server.WriteSRT.write_calls == []  # WhisperX writer not invoked
    assert "-->" in resp.content  # existing EC2 assertion still holds
    assert resp.media_type == "application/x-subrip"


def test_format_response_vtt_without_knob_uses_legacy():
    """No knob on vtt keeps the legacy _to_vtt output starting with WEBVTT."""
    server = _load_server()
    resp = server._format_response(_result(), "vtt", want_words=False, diarize=False)
    assert server.WriteVTT.write_calls == []
    assert resp.content.startswith("WEBVTT")  # existing EC2 assertion still holds
    assert resp.media_type == "text/vtt"


def test_format_response_json_ignores_knobs():
    """json + knobs returns normally; the WhisperX writer is never invoked."""
    server = _load_server()
    resp = server._format_response(
        _result(),
        "json",
        want_words=False,
        diarize=False,
        max_line_width=42,
        max_line_count=2,
        highlight_words=True,
    )
    assert server.WriteSRT.write_calls == []
    assert server.WriteVTT.write_calls == []
    assert resp.content == {"text": "hello world"}


def test_format_response_text_ignores_knobs():
    """text + knobs returns the plain transcript; no WhisperX writer call."""
    server = _load_server()
    resp = server._format_response(
        _result(), "text", want_words=False, diarize=False, highlight_words=True
    )
    assert server.WriteSRT.write_calls == []
    assert server.WriteVTT.write_calls == []
    assert resp.content == "hello world"


# ---------------------------------------------------------------------------
# _handle_transcription — subtitle knob forces alignment (want_words) on srt/vtt
# ---------------------------------------------------------------------------
class _FakeUpload:
    """One-shot upload: yields the payload once, then b"" to end the chunk loop.

    _handle_transcription streams the body via ``await file.read(n)``, so the
    fake accepts the chunk-size arg and signals EOF on the second call.
    """

    filename = "audio.wav"

    def __init__(self):
        self._sent = False

    async def read(self, size=-1):
        if self._sent:
            return b""
        self._sent = True
        return b"RIFFfake-audio-bytes"


def _run_handle(server, *, response_format, timestamp_granularities=None, **knobs):
    """Drive _handle_transcription with _transcribe stubbed to record want_words."""
    recorded: dict = {}

    def _recorder(**kwargs):
        recorded["want_words"] = kwargs["want_words"]
        return _result()

    server._transcribe = _recorder
    resp = asyncio.run(
        server._handle_transcription(
            file=_FakeUpload(),
            language=None,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            **knobs,
        )
    )
    return recorded, resp


def test_handle_transcription_srt_knob_forces_alignment():
    """srt + max_line_width forces want_words=True even without granularities[]=word."""
    server = _load_server()
    recorded, resp = _run_handle(server, response_format="srt", max_line_width=42)
    assert recorded["want_words"] is True
    assert resp.content == _SRT_SENTINEL  # routed through the WhisperX writer


def test_handle_transcription_vtt_highlight_forces_alignment():
    """vtt + highlight_words forces alignment too."""
    server = _load_server()
    recorded, _ = _run_handle(server, response_format="vtt", highlight_words=True)
    assert recorded["want_words"] is True


def test_handle_transcription_srt_no_knob_does_not_force_alignment():
    """srt with no knob leaves want_words as derived (False here)."""
    server = _load_server()
    recorded, resp = _run_handle(server, response_format="srt")
    assert recorded["want_words"] is False
    assert server.WriteSRT.write_calls == []  # legacy formatter path


def test_handle_transcription_json_knob_does_not_force_alignment():
    """json + max_line_width: knob ignored, no alignment forced, no error."""
    server = _load_server()
    recorded, resp = _run_handle(server, response_format="json", max_line_width=42)
    assert recorded["want_words"] is False
    assert resp.content == {"text": "hello world"}  # returns normally
    assert server.WriteSRT.write_calls == []
    assert server.WriteVTT.write_calls == []


def test_handle_transcription_word_granularity_preserved_with_knob():
    """subtitle_formatting ORs into want_words; explicit word granularity is kept."""
    server = _load_server()
    recorded, _ = _run_handle(
        server,
        response_format="srt",
        timestamp_granularities=["word"],
        max_line_width=42,
    )
    assert recorded["want_words"] is True


def test_handle_transcription_word_granularity_without_knob():
    """granularities[]=word still forces want_words even when no subtitle knob is set."""
    server = _load_server()
    recorded, _ = _run_handle(server, response_format="json", timestamp_granularities=["word"])
    assert recorded["want_words"] is True
