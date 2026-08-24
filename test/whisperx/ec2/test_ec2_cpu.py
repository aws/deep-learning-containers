# no-op: verify the whisperx/ec2 test-skip cache (record + hit) against the prod image
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
"""End-to-end EC2 test that runs the WhisperX GPU image on a CPU-only host.

The container is started with NO `--gpus` flag to force CPU execution, which
validates that it still starts and serves on CPU — the regression guarded by the
start_cuda_compat.sh fix — and that real transcription works on CPU.

Assertions are deliberately loose about ASR *content* (ASR output is
nondeterministic): we assert the response contract — status codes, response
shapes, and timestamp structure — not exact transcript strings. Container
lifecycle lives in common.py; this file only sets device config and the CPU test
cases. The suite is kept bounded (no diarization / response-format / error
cases) so it stays fast on a CPU runner.
"""

import requests
from whisperx.ec2.common import (
    AUDIO_EN,
    download_fixture,
    make_container_fixture,
    post_transcription,
)

DEVICE = "cpu"
DOCKER_RUN_FLAGS = None  # no --gpus: force CPU

# Register the container fixture for this module (function-scoped: fresh
# container per test, mirroring the GPU module). tiny keeps CPU warm-load +
# inference fast enough for CI.
container = make_container_fixture(
    DEVICE, docker_run_flags=DOCKER_RUN_FLAGS, env={"WHISPERX_DEFAULT_MODEL": "tiny"}
)


def test_ping_and_models(container):
    """/ping is healthy and /v1/models advertises at least one served model id.

    Core e2e proof that the container STARTS and serves on CPU (no --gpus, no
    S3/audio needed).
    """
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
    """Default json transcription of English audio returns non-empty text on CPU."""
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))

    resp = post_transcription(container["port"], audio, response_format="json")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    # json response shape is exactly {"text": "..."}.
    assert body.get("text", "").strip(), "expected non-empty transcription text"


def test_word_timestamps(container, aws_session, tmp_path):
    """verbose_json + timestamp_granularities[]=word yields coherent word timings on CPU."""
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
