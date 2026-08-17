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
"""EC2 model-launch-config tests for the WhisperX ASR DLC (GPU).

Group A: these tests validate how the server's launch-time env vars shape the
served-model contract — WHISPERX_DEFAULT_MODEL (the one model per container) and
WHISPERX_SERVED_MODEL_NAME (a client-facing alias advertised by /v1/models).
Each case needs a container launched with different env, so unlike test_ec2_gpu.py
they do NOT use the shared default `container` fixture; instead each spins up (and
tears down) a custom-env container via common.run_container_with_env.

The Whisper model is one-per-container: the request `model` field is accepted but
ignored (an OpenAI-compat no-op), so every transcription runs the launched model
regardless of the `model` value sent.

WHISPERX_DEFAULT_MODEL=tiny is used throughout so the startup warm-load finishes
in seconds — tiny is a valid faster-whisper model that downloads far faster than
the large-v2 default.

Assertions target the served contract only (advertised model id, and that any
`model` value transcribes 200) — never exact transcript text, which is
nondeterministic.
"""

import requests
from whisperx.ec2.common import (
    AUDIO_EN,
    download_fixture,
    post_transcription,
    run_container_with_env,
)

DEVICE = "gpu"
DOCKER_RUN_FLAGS = ["--gpus", "all"]
# Small, fast-downloading Whisper model so the startup warm-load takes seconds,
# not the minutes large-v2 needs. tiny is a valid faster-whisper id.
FAST_MODEL = "tiny"


def test_custom_default_model(image_uri, aws_session, tmp_path):
    """WHISPERX_DEFAULT_MODEL sets the one served model; the `model` field is ignored.

    Launches with WHISPERX_DEFAULT_MODEL=tiny and asserts /v1/models advertises
    `tiny`, that omitting `model` and naming `tiny` both transcribe (200), and
    that naming a *different* id (`large-v2`) also transcribes 200 — the field is
    an ignored no-op, so the launched model runs regardless.
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    with run_container_with_env(
        image_uri, {"WHISPERX_DEFAULT_MODEL": FAST_MODEL}, device=DEVICE, flags=DOCKER_RUN_FLAGS
    ) as c:
        port = c["port"]

        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data[0]["id"] == FAST_MODEL, f"expected served id {FAST_MODEL!r}, got {data}"

        # Omitted model -> serves the launched model -> 200 with real text.
        resp = post_transcription(port, audio, response_format="json")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("text", "").strip(), "expected non-empty text with default model"

        # Naming the launched model explicitly -> 200.
        resp = post_transcription(port, audio, model=FAST_MODEL, response_format="json")
        assert resp.status_code == 200, resp.text

        # A different id -> still 200: the `model` field is ignored, so the
        # launched model serves the request instead of being rejected.
        resp = post_transcription(port, audio, model="large-v2", response_format="json")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("text", "").strip(), "expected launched model to serve any model id"


def test_served_model_alias(image_uri, aws_session, tmp_path):
    """WHISPERX_SERVED_MODEL_NAME sets the id /v1/models advertises for OpenAI clients.

    Launches tiny behind the alias `whisper-1` and asserts /v1/models reports
    `whisper-1`, and that the alias, the real backing id `tiny`, and an unrelated
    id all transcribe (200) — the `model` field is an ignored no-op, so every
    request runs the launched model.
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    env = {"WHISPERX_DEFAULT_MODEL": FAST_MODEL, "WHISPERX_SERVED_MODEL_NAME": "whisper-1"}
    with run_container_with_env(image_uri, env, device=DEVICE, flags=DOCKER_RUN_FLAGS) as c:
        port = c["port"]

        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data[0]["id"] == "whisper-1", f"expected alias id 'whisper-1', got {data}"

        # The advertised alias transcribes -> 200.
        resp = post_transcription(port, audio, model="whisper-1", response_format="json")
        assert resp.status_code == 200, resp.text

        # The real backing model id transcribes too -> 200.
        resp = post_transcription(port, audio, model=FAST_MODEL, response_format="json")
        assert resp.status_code == 200, resp.text

        # An unrelated id -> still 200: the `model` field is ignored.
        resp = post_transcription(port, audio, model="whisper-large", response_format="json")
        assert resp.status_code == 200, resp.text
