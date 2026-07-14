# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""EC2 container integration tests for the WhisperX ASR DLC (GPU).

Runs the container locally via `docker run --gpus all`, waits for the FastAPI
server on port 8000 to warm-load its models, and validates the OpenAI-compatible
transcription contract against staged audio fixtures.

Assertions are deliberately loose about ASR *content* (ASR output is
nondeterministic): we assert the response contract — status codes, response
shapes, content types, and timestamp/speaker structure — not exact transcript
strings. Container lifecycle lives in common.py; this file only sets device
config and the test cases.
"""

import requests
from whisperx.ec2.common import (
    AUDIO_DIARIZE_2SPK,
    AUDIO_EN,
    AUDIO_ZH,
    download_fixture,
    make_container_fixture,
    post_transcription,
)

DEVICE = "gpu"
DOCKER_RUN_FLAGS = ["--gpus", "all"]

# Register the container fixture for this module (function-scoped: fresh
# container per test, mirroring the Ray EC2 suite).
container = make_container_fixture(DEVICE, docker_run_flags=DOCKER_RUN_FLAGS)


def test_ping_and_models(container):
    """/ping is healthy and /v1/models advertises at least one served model id."""
    port = container["port"]

    resp = requests.get(f"http://localhost:{port}/ping", timeout=10)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("object") == "list"
    data = body.get("data")
    assert isinstance(data, list) and len(data) >= 1, f"expected >=1 model, got {body}"
    assert data[0].get("id"), "expected a non-empty model id"


def test_basic_transcription(container, aws_session, tmp_path):
    """Default json transcription of English audio returns non-empty text."""
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))

    resp = post_transcription(container["port"], audio, response_format="json")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    # json response shape is exactly {"text": "..."}.
    assert body.get("text", "").strip(), "expected non-empty transcription text"


def test_word_timestamps(container, aws_session, tmp_path):
    """verbose_json + timestamp_granularities[]=word yields coherent word timings."""
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))

    resp = post_transcription(
        container["port"],
        audio,
        response_format="verbose_json",
        # The server reads the list off the literal "timestamp_granularities[]" key.
        **{"timestamp_granularities[]": ["word"]},
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    words = body.get("words")
    assert isinstance(words, list) and words, f"expected non-empty words list, got {body.keys()}"
    for w in words:
        assert {"word", "start", "end"} <= set(w), f"word entry missing keys: {w}"

    # WhisperX alignment can leave a few tokens (e.g. digits) without timings;
    # assert coherence on the words that ARE timed and require at least one.
    timed = [
        (w["start"], w["end"]) for w in words if w["start"] is not None and w["end"] is not None
    ]
    assert timed, "expected at least one word with start/end timestamps"
    for start, end in timed:
        assert isinstance(start, (int, float)) and isinstance(end, (int, float))
        assert start <= end, f"word start {start} after end {end}"


def test_language_non_english(container, aws_session, tmp_path):
    """Forcing language=zh transcribes Chinese audio and echoes the language."""
    audio = download_fixture(aws_session, AUDIO_ZH, str(tmp_path / AUDIO_ZH))

    resp = post_transcription(
        container["port"],
        audio,
        language="zh",
        response_format="verbose_json",
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body.get("language") == "zh", f"expected language 'zh', got {body.get('language')!r}"
    assert body.get("text", "").strip(), "expected non-empty transcription text"


def test_diarization(container, aws_session, tmp_path):
    """diarize=true on a 2-speaker clip returns >=2 speakers and per-segment labels."""
    audio = download_fixture(aws_session, AUDIO_DIARIZE_2SPK, str(tmp_path / AUDIO_DIARIZE_2SPK))

    resp = post_transcription(
        container["port"],
        audio,
        diarize=True,
        response_format="verbose_json",
    )
    assert resp.status_code == 200, resp.text

    body = resp.json()
    speakers = body.get("speakers")
    assert isinstance(speakers, list), f"expected speakers list, got {body.keys()}"
    assert len(set(speakers)) >= 2, f"expected >=2 distinct speakers, got {speakers}"

    segments = body.get("segments", [])
    seg_speakers = [s.get("speaker") for s in segments if s.get("speaker")]
    assert seg_speakers, "expected at least one segment to carry a speaker label"


def test_response_formats(container, aws_session, tmp_path):
    """text / srt / vtt each return the right content-type and structure."""
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    port = container["port"]

    resp = post_transcription(port, audio, response_format="text")
    assert resp.status_code == 200, resp.text
    assert "text/plain" in resp.headers.get("content-type", "")
    assert resp.text.strip(), "expected non-empty plain text"

    resp = post_transcription(port, audio, response_format="srt")
    assert resp.status_code == 200, resp.text
    assert "application/x-subrip" in resp.headers.get("content-type", "")
    assert "-->" in resp.text, "expected SRT cue timing arrow"

    resp = post_transcription(port, audio, response_format="vtt")
    assert resp.status_code == 200, resp.text
    assert "text/vtt" in resp.headers.get("content-type", "")
    assert resp.text.startswith("WEBVTT"), "expected VTT header"


def test_errors(container, tmp_path):
    """Contract errors: unknown model -> 404, empty upload -> 400, bad format -> 400.

    The server validates the model, then the response_format, then reads the
    file, so the model/format cases need only a present (unread) file part.
    """
    port = container["port"]

    # A present-but-never-read file part (validation short-circuits before read).
    dummy = tmp_path / "dummy.wav"
    dummy.write_bytes(b"RIFF0000WAVE")

    resp = post_transcription(port, str(dummy), model="does-not-exist-xyz")
    assert resp.status_code == 404, resp.text

    resp = post_transcription(port, str(dummy), response_format="xml")
    assert resp.status_code == 400, resp.text

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    resp = post_transcription(port, str(empty), response_format="json")
    assert resp.status_code == 400, resp.text
