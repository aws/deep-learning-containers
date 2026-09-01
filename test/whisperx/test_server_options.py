"""Unit tests for the WhisperX server's launch-time option builders.

These run on CPU with no container and no GPU: torch / whisperx / fastapi are
stubbed in ``sys.modules`` before ``server.py`` is imported, so importing the
server never pulls in CUDA, the whisperX wheel, or a real ASGI app. We then
exercise the pure ``env -> dict`` helpers (``_build_asr_options`` /
``_build_vad_options`` / ``_env_bool``) that translate WHISPERX_* launch env
into ``whisperx.load_model`` kwargs, plus the module-level constants they feed.
A few tests go further and assert those constants are actually forwarded into
the ``whisperx`` calls: ``_get_align`` pins the aligner via ``model_name`` and
``_transcribe`` passes ``task`` (and no stray decoding kwargs) to transcribe().
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# server.py lives in the image-build tree, a sibling of test/; load it by path.
SERVER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "docker" / "whisperx" / "server.py"

# Every WHISPERX_* var the option builders read. Cleared before each test so a
# stray value in the runner's environment can't skew the "defaults" assertions.
_WHISPERX_ENV = (
    "WHISPERX_TEMPERATURE",
    "WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK",
    "WHISPERX_BEAM_SIZE",
    "WHISPERX_BEST_OF",
    "WHISPERX_PATIENCE",
    "WHISPERX_LENGTH_PENALTY",
    "WHISPERX_COMPRESSION_RATIO_THRESHOLD",
    "WHISPERX_LOGPROB_THRESHOLD",
    "WHISPERX_NO_SPEECH_THRESHOLD",
    "WHISPERX_CONDITION_ON_PREVIOUS_TEXT",
    "WHISPERX_INITIAL_PROMPT",
    "WHISPERX_HOTWORDS",
    "WHISPERX_SUPPRESS_TOKENS",
    "WHISPERX_SUPPRESS_NUMERALS",
    "WHISPERX_CHUNK_SIZE",
    "WHISPERX_VAD_ONSET",
    "WHISPERX_VAD_OFFSET",
    "WHISPERX_VAD_METHOD",
    "WHISPERX_TASK",
    "WHISPERX_ALIGN_MODEL",
)


def _install_stubs() -> None:
    """Register lightweight fakes for the heavy imports server.py performs.

    server.py runs ``torch.cuda.is_available()`` and builds a FastAPI app at
    import time; none of that is relevant to the option builders, so we swap the
    modules for minimal stand-ins that satisfy the import and the decorators.
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

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeApp
    fastapi.File = _form_marker
    fastapi.Form = _form_marker
    fastapi.UploadFile = type("UploadFile", (), {})
    fastapi.HTTPException = type("HTTPException", (Exception,), {})

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.JSONResponse = type("JSONResponse", (), {})
    fastapi_responses.PlainTextResponse = type("PlainTextResponse", (), {})

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
    """Import server.py fresh (stubs installed) so it re-reads WHISPERX_* env."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location("whisperx_server_under_test", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_server()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from a WHISPERX_*-free environment."""
    for name in _WHISPERX_ENV:
        monkeypatch.delenv(name, raising=False)


def test_asr_options_defaults():
    assert server._build_asr_options() == {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperatures": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "hotwords": None,
        "suppress_tokens": [-1],
        "suppress_numerals": False,
    }


def test_vad_options_defaults():
    assert server._build_vad_options() == {
        "chunk_size": 30,
        "vad_onset": 0.5,
        "vad_offset": 0.363,
    }


def test_beam_size_override(monkeypatch):
    monkeypatch.setenv("WHISPERX_BEAM_SIZE", "8")
    assert server._build_asr_options()["beam_size"] == 8


def test_vad_onset_override(monkeypatch):
    monkeypatch.setenv("WHISPERX_VAD_ONSET", "0.7")
    assert server._build_vad_options()["vad_onset"] == 0.7


def test_suppress_tokens_csv(monkeypatch):
    monkeypatch.setenv("WHISPERX_SUPPRESS_TOKENS", "1,2,3")
    assert server._build_asr_options()["suppress_tokens"] == [1, 2, 3]


def test_suppress_tokens_ignores_blanks(monkeypatch):
    monkeypatch.setenv("WHISPERX_SUPPRESS_TOKENS", "1, ,3,")
    assert server._build_asr_options()["suppress_tokens"] == [1, 3]


def test_suppress_numerals_truthy(monkeypatch):
    monkeypatch.setenv("WHISPERX_SUPPRESS_NUMERALS", "true")
    assert server._build_asr_options()["suppress_numerals"] is True


def test_temperature_fallback_schedule(monkeypatch):
    monkeypatch.setenv("WHISPERX_TEMPERATURE", "0.4")
    # WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK defaults to 0.2.
    assert server._build_asr_options()["temperatures"] == (0.4, 0.6, 0.8, 1.0)


def test_temperature_increment_disabled(monkeypatch):
    monkeypatch.setenv("WHISPERX_TEMPERATURE", "0.4")
    monkeypatch.setenv("WHISPERX_TEMPERATURE_INCREMENT_ON_FALLBACK", "0")
    assert server._build_asr_options()["temperatures"] == (0.4,)


def test_initial_prompt_set(monkeypatch):
    monkeypatch.setenv("WHISPERX_INITIAL_PROMPT", "medical dictation")
    assert server._build_asr_options()["initial_prompt"] == "medical dictation"


def test_initial_prompt_empty_is_none(monkeypatch):
    monkeypatch.setenv("WHISPERX_INITIAL_PROMPT", "")
    assert server._build_asr_options()["initial_prompt"] is None


def test_condition_on_previous_text_toggle(monkeypatch):
    monkeypatch.setenv("WHISPERX_CONDITION_ON_PREVIOUS_TEXT", "yes")
    assert server._build_asr_options()["condition_on_previous_text"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nonsense", False),
    ],
)
def test_env_bool(monkeypatch, value, expected):
    monkeypatch.setenv("WHISPERX_SUPPRESS_NUMERALS", value)
    assert server._env_bool("WHISPERX_SUPPRESS_NUMERALS") is expected


def test_env_bool_default_when_unset():
    assert server._env_bool("WHISPERX_DOES_NOT_EXIST") is False
    assert server._env_bool("WHISPERX_DOES_NOT_EXIST", default=True) is True


def test_module_constants_reflect_env(monkeypatch):
    """The module-level ASR/VAD constants are built from env at import time."""
    monkeypatch.setenv("WHISPERX_BEAM_SIZE", "9")
    monkeypatch.setenv("WHISPERX_CHUNK_SIZE", "12")
    reloaded = _load_server()
    assert reloaded.ASR_OPTIONS["beam_size"] == 9
    assert reloaded.VAD_OPTIONS["chunk_size"] == 12


def test_pipeline_constants_override(monkeypatch):
    monkeypatch.setenv("WHISPERX_VAD_METHOD", "silero")
    monkeypatch.setenv("WHISPERX_TASK", "translate")
    monkeypatch.setenv("WHISPERX_ALIGN_MODEL", "WAV2VEC2_ASR_LARGE_LV60K_960H")
    reloaded = _load_server()
    assert reloaded.VAD_METHOD == "silero"
    assert reloaded.TASK == "translate"
    assert reloaded.ALIGN_MODEL == "WAV2VEC2_ASR_LARGE_LV60K_960H"


def test_pipeline_constants_defaults(monkeypatch):
    # Reload under a cleaned env instead of trusting the collection-time module:
    # a runner with WHISPERX_TASK/VAD_METHOD/ALIGN_MODEL exported must not break
    # the defaults assertions.
    for name in ("WHISPERX_VAD_METHOD", "WHISPERX_TASK", "WHISPERX_ALIGN_MODEL"):
        monkeypatch.delenv(name, raising=False)
    reloaded = _load_server()
    assert reloaded.VAD_METHOD == "pyannote"
    assert reloaded.TASK == "transcribe"
    assert reloaded.ALIGN_MODEL is None


def test_get_align_forwards_default_model_name(monkeypatch):
    """Unset WHISPERX_ALIGN_MODEL => _get_align passes model_name=None (default aligner)."""
    monkeypatch.delenv("WHISPERX_ALIGN_MODEL", raising=False)
    reloaded = _load_server()
    assert reloaded.ALIGN_MODEL is None

    recorded: dict = {}

    def _fake_load_align_model(*args, **kwargs):
        recorded.update(kwargs)
        return object(), {}

    reloaded.whisperx.load_align_model = _fake_load_align_model
    reloaded._get_align("en")
    assert recorded["model_name"] is None


def test_get_align_forwards_pinned_model_name(monkeypatch):
    """WHISPERX_ALIGN_MODEL set => _get_align forwards it as model_name to pin the aligner."""
    monkeypatch.setenv("WHISPERX_ALIGN_MODEL", "WAV2VEC2_ASR_LARGE_LV60K_960H")
    reloaded = _load_server()
    assert reloaded.ALIGN_MODEL == "WAV2VEC2_ASR_LARGE_LV60K_960H"

    recorded: dict = {}

    def _fake_load_align_model(*args, **kwargs):
        recorded.update(kwargs)
        return object(), {}

    reloaded.whisperx.load_align_model = _fake_load_align_model
    # Fresh reload => empty _ALIGN_LRU, so the stub is actually invoked.
    reloaded._get_align("de")
    assert recorded["model_name"] == "WAV2VEC2_ASR_LARGE_LV60K_960H"


def test_transcribe_forwards_task_and_no_decoding_kwargs(monkeypatch):
    """Regression for the inert-knob bug: transcribe() receives only pipeline kwargs.

    ``task`` must reach ``FasterWhisperPipeline.transcribe`` (load_model discards
    it without a pinned language), and decoding kwargs baked into ASR_OPTIONS
    (initial_prompt/temperature/...) must NOT leak into the per-call kwargs.
    """
    monkeypatch.setenv("WHISPERX_TASK", "translate")
    reloaded = _load_server()

    recorded: dict = {}

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            recorded.update(kwargs)
            return {"segments": [{"text": "hi", "start": 0.0, "end": 1.0}], "language": "en"}

    reloaded.whisperx.load_audio = lambda path: [0.0] * 16000
    reloaded._get_whisper = lambda name: _FakeModel()

    result = reloaded._transcribe(
        audio_path="/tmp/does-not-exist.wav",
        language=None,
        want_words=False,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
    )

    assert set(recorded).issubset({"batch_size", "chunk_size", "language", "task"})
    assert "initial_prompt" not in recorded
    assert "temperature" not in recorded
    assert recorded["task"] == "translate"
    assert recorded["task"] == reloaded.TASK
    assert result["segments"]  # sanity: the pipeline still produced output
